package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ollama/ollama/api"
)

// config holds all benchmark configuration.
type config struct {
	binary         string
	models         []string
	contexts       []int
	kvModes        []string
	epochs         int
	warmup         int
	predict        int
	maxPromptTokens int // cap on prompt length (0 = contextSize - predict - 64)
	port           int
	outputDir      string
	timeout        time.Duration
	cudaDevice     string
	modelsDir      string // OLLAMA_MODELS override
	debug          bool
	tqFusedOff     bool   // set OLLAMA_TQ_FUSED=0 on server (disable fused kernel)
	flashAttn      bool   // OLLAMA_FLASH_ATTENTION value passed to the spawned server
	ppl            bool   // collect per-token logprobs and compute decode PPL
	commit         string // git commit hash to stamp in output files
}

// cellResult holds the outcome of one (model, kvMode, contextSize, epoch) cell.
type cellResult struct {
	Model          string
	KVMode         string
	ContextSize    int
	Epoch          int
	PrefillTokens  int
	PrefillTokSec  float64
	DecodeTokens   int
	DecodeTokSec   float64
	PPL            float64 // perplexity of decode tokens (0 if not collected)
	KVCacheMB      float64 // estimated from architecture
	SizeVRAMMB     float64 // from /api/ps
	SizeTotalMB    float64 // from /api/ps
	FullyOffloaded bool
	BlockCount     int
	EstGPULayers   int
	FitStatus      string // "ok", "partial_offload", "oom", "error"
	Error          string
}

// modelInfo holds architecture metadata extracted from /api/show.
type modelInfo struct {
	Family      string
	Params      string
	Quant       string
	BlockCount  int
	HeadDimK    int
	HeadDimV    int
	HeadCountKV int
}

// Word list for prompt generation (~90 words, cycling).
var wordList = []string{
	"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
	"a", "bright", "sunny", "day", "in", "meadow", "where",
	"flowers", "bloom", "and", "birds", "sing", "their", "morning",
	"songs", "while", "gentle", "breeze", "carries", "sweet", "scent",
	"of", "pine", "trees", "across", "rolling", "hills", "toward",
	"distant", "mountains", "covered", "with", "fresh", "snow",
	"beneath", "clear", "blue", "sky", "children", "play", "near",
	"old", "stone", "bridge", "that", "crosses", "winding", "river",
	"water", "flows", "silently", "past", "green", "banks", "tall",
	"grass", "sways", "wind", "clouds", "drift", "slowly", "above",
	"forest", "dense", "shadows", "cool", "shade", "path", "leads",
	"deeper", "into", "wilderness", "animals", "roam", "freely",
	"here", "life", "abundant", "rich", "full",
}

// generatePrompt creates a repeating-word prompt targeting approximately targetTokens tokens.
// epoch is used to vary the starting offset, defeating KV cache prefix matching.
func generatePrompt(targetTokens int, epoch int) string {
	targetWords := int(float64(targetTokens) / 1.3)
	if targetWords < 1 {
		targetWords = 1
	}
	offset := epoch * 7
	n := len(wordList)
	words := make([]string, targetWords)
	for i := range words {
		words[i] = wordList[((i+offset)%n+n)%n]
	}
	return strings.Join(words, " ")
}

// kBytesPerElement returns bytes per K element for the given mode.
func kBytesPerElement(kvMode string, headDim int) float64 {
	switch kvMode {
	case "tq4", "tq4k":
		// 4 bits packed + 1 f32 scale per headDim elements
		return float64(4)/8 + float64(4)/float64(headDim)
	case "tq3", "tq3k":
		// 3 bits packed + 1 f32 scale per headDim elements
		return float64(3)/8 + float64(4)/float64(headDim)
	case "tq2", "tq2k":
		// 2 bits packed + 1 f32 scale per headDim elements
		return float64(2)/8 + float64(4)/float64(headDim)
	case "q8_0":
		return 1
	case "q4_0":
		return 0.5
	case "f32":
		return 4
	default: // f16
		return 2
	}
}

// vBytesPerElement returns bytes per V element for the given mode.
// K-only presets (tq2k/tq3k) keep V as f16.
func vBytesPerElement(kvMode string, headDim int) float64 {
	if kvMode == "tq2k" || kvMode == "tq3k" || kvMode == "tq4k" {
		return 2 // f16
	}
	return kBytesPerElement(kvMode, headDim)
}

// estimateKVCacheMB estimates the KV cache memory in MB.
func estimateKVCacheMB(info modelInfo, contextSize int, kvMode string) float64 {
	if info.BlockCount == 0 || info.HeadCountKV == 0 {
		return 0
	}
	headDimK := info.HeadDimK
	if headDimK == 0 {
		headDimK = 128
	}
	headDimV := info.HeadDimV
	if headDimV == 0 {
		headDimV = 128
	}
	kBpe := kBytesPerElement(kvMode, headDimK)
	vBpe := vBytesPerElement(kvMode, headDimV)
	totalBytes := float64(contextSize) *
		float64(info.HeadCountKV) *
		float64(info.BlockCount) *
		(float64(headDimK)*kBpe + float64(headDimV)*vBpe)
	return totalBytes / (1024 * 1024)
}

// fetchModelInfo retrieves architecture metadata for a model via /api/show.
// GGUF metadata keys are prefixed by architecture (e.g. "llama.", "qwen2.", "qwen3moe."),
// so we scan all keys by suffix rather than hardcoding prefixes.
func fetchModelInfo(ctx context.Context, client *api.Client, model string) modelInfo {
	info := modelInfo{
		HeadDimK:    128,
		HeadDimV:    128,
		HeadCountKV: 8,
		BlockCount:  32,
	}

	resp, err := client.Show(ctx, &api.ShowRequest{Model: model})
	if err != nil {
		fmt.Fprintf(os.Stderr, "WARNING: Could not fetch model info for '%s': %v\n", model, err)
		return info
	}

	info.Family = resp.Details.Family
	info.Params = resp.Details.ParameterSize
	info.Quant = resp.Details.QuantizationLevel

	mi := resp.ModelInfo
	if mi == nil {
		return info
	}

	// Scan all keys by suffix — GGUF uses arch-prefixed keys like
	// "llama.block_count", "qwen3moe.attention.key_length", etc.
	miFloat := func(suffix string) (int, bool) {
		for k, v := range mi {
			if strings.HasSuffix(k, suffix) {
				// Skip vision sub-model keys
				if strings.Contains(k, "vision") {
					continue
				}
				if f, ok := v.(float64); ok {
					return int(f), true
				}
			}
		}
		return 0, false
	}

	if v, ok := miFloat(".block_count"); ok {
		info.BlockCount = v
	}
	if v, ok := miFloat(".attention.key_length"); ok {
		info.HeadDimK = v
	}
	if v, ok := miFloat(".attention.value_length"); ok {
		info.HeadDimV = v
	}
	if v, ok := miFloat(".attention.head_count_kv"); ok {
		info.HeadCountKV = v
	}

	return info
}

// startServer launches an ollama server process with the given KV mode.
func startServer(binary, kvMode string, port int, cudaDevice, modelsDir string, debug, tqFusedOff, flashAttn bool) (*exec.Cmd, error) {
	cmd := exec.Command(binary, "serve")

	// Build filtered environment: strip keys we control, pass everything else.
	// HSA_OVERRIDE_GFX_VERSION is also stripped: when set it causes Ollama's
	// NVML-based GPU discovery to malfunction, falling back to CPU instead of
	// picking the best available CUDA device.
	env := make([]string, 0, len(os.Environ())+6)
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "OLLAMA_HOST=") ||
			strings.HasPrefix(e, "OLLAMA_KV_CACHE_TYPE=") ||
			strings.HasPrefix(e, "OLLAMA_FLASH_ATTENTION=") ||
			strings.HasPrefix(e, "OLLAMA_NEW_ENGINE=") ||
			strings.HasPrefix(e, "OLLAMA_MODELS=") ||
			strings.HasPrefix(e, "HSA_OVERRIDE_GFX_VERSION=") {
			continue
		}
		env = append(env, e)
	}

	faVal := "1"
	if !flashAttn {
		faVal = "0"
	}
	env = append(env,
		fmt.Sprintf("OLLAMA_HOST=127.0.0.1:%d", port),
		fmt.Sprintf("OLLAMA_FLASH_ATTENTION=%s", faVal),
		// TurboQuant only works through the Ollama native engine. Without
		// this, models with a llama.cpp code path (llama, qwen, ...) fall
		// back to the legacy runner and silently bypass TQ — producing
		// identical numbers to f16 across all TQ modes.
		"OLLAMA_NEW_ENGINE=true",
	)
	// Only set CUDA_VISIBLE_DEVICES when explicitly requested (ollama uses
	// NVML for GPU discovery which ignores this var; setting it can cause
	// NVML/CUDA index mismatches and wrong GPU selection).
	if cudaDevice != "" {
		env = append(env, fmt.Sprintf("CUDA_VISIBLE_DEVICES=%s", cudaDevice))
	}
	if modelsDir != "" {
		env = append(env, fmt.Sprintf("OLLAMA_MODELS=%s", modelsDir))
	}
	if kvMode != "" && kvMode != "f16" {
		env = append(env, fmt.Sprintf("OLLAMA_KV_CACHE_TYPE=%s", kvMode))
	}
	_ = tqFusedOff // retained for CLI flag compatibility; fused-kernel gating removed

	cmd.Env = env

	if debug {
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("startServer: %w", err)
	}
	return cmd, nil
}

// waitForHealth polls the ollama server health endpoint until it responds or timeout.
func waitForHealth(port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://127.0.0.1:%d/", port)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url) //nolint:gosec
		if err == nil {
			resp.Body.Close()
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("server on port %d did not become healthy within %s", port, timeout)
}

// stopServer sends SIGTERM and waits up to 10s before force-killing.
func stopServer(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Signal(syscall.SIGTERM)

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		_ = cmd.Process.Kill()
		<-done
	}
}

// makeClient creates an api.Client pointed at the given port.
func makeClient(port int) (*api.Client, error) {
	base, err := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", port))
	if err != nil {
		return nil, err
	}
	return api.NewClient(base, http.DefaultClient), nil
}

// benchmarkCell runs a single generate request and returns token counts, durations, and PPL.
// When ppl is true, logprobs are collected and perplexity computed from the decode tokens.
// Returns prefillTok, prefillDur, decodeTok, decodeDur, ppl or an error.
func benchmarkCell(
	client *api.Client,
	model string,
	contextSize, predict, maxPromptTokens, epoch int,
	timeout time.Duration,
	debug, ppl bool,
) (prefillTok int, prefillDur time.Duration, decodeTok int, decodeDur time.Duration, cellPPL float64, err error) {

	promptTokens := contextSize - predict - 64
	defaultCap := 16384
	if maxPromptTokens > 0 {
		defaultCap = maxPromptTokens
	}
	if promptTokens > defaultCap {
		promptTokens = defaultCap
	}
	if promptTokens < 64 {
		promptTokens = 64
	}

	prompt := generatePrompt(promptTokens, epoch)

	options := map[string]any{
		"num_ctx":     contextSize,
		"num_predict": predict,
		"temperature": 0,
		"seed":        42,
	}

	req := &api.GenerateRequest{
		Model:    model,
		Prompt:   prompt,
		Raw:      true,
		Options:  options,
		Logprobs: ppl,
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	var metrics *api.Metrics
	var logprobSum float64
	var logprobCount int
	err = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		if debug {
			fmt.Fprint(os.Stderr, resp.Response)
		}
		for _, lp := range resp.Logprobs {
			logprobSum += lp.Logprob
			logprobCount++
		}
		if resp.Done {
			metrics = &resp.Metrics
		}
		return nil
	})
	if err != nil {
		return 0, 0, 0, 0, 0, err
	}
	if metrics == nil {
		return 0, 0, 0, 0, 0, fmt.Errorf("no metrics received")
	}

	if ppl && logprobCount > 0 {
		cellPPL = math.Exp(-logprobSum / float64(logprobCount))
	}

	return metrics.PromptEvalCount, metrics.PromptEvalDuration,
		metrics.EvalCount, metrics.EvalDuration, cellPPL, nil
}

// fetchVRAM returns (sizeVRAM, sizeTotal) for the named model from /api/ps.
func fetchVRAM(ctx context.Context, client *api.Client, model string) (sizeVRAM, sizeTotal int64) {
	resp, err := client.ListRunning(ctx)
	if err != nil {
		return 0, 0
	}
	// Exact match first
	for _, m := range resp.Models {
		if m.Name == model || m.Model == model {
			return m.SizeVRAM, m.Size
		}
	}
	// Prefix match (handles :latest tags etc.)
	for _, m := range resp.Models {
		if strings.HasPrefix(m.Name, model) || strings.HasPrefix(m.Model, model) {
			return m.SizeVRAM, m.Size
		}
	}
	// Substring match (handles registry-prefixed or digest names)
	for _, m := range resp.Models {
		if strings.Contains(m.Name, model) || strings.Contains(m.Model, model) {
			return m.SizeVRAM, m.Size
		}
	}
	return 0, 0
}

// unloadModel sends a zero keep-alive generate to evict the model from memory.
func unloadModel(client *api.Client, model string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	zero := api.Duration{Duration: 0}
	req := &api.GenerateRequest{
		Model:     model,
		KeepAlive: &zero,
	}
	_ = client.Generate(ctx, req, func(resp api.GenerateResponse) error { return nil })
}

// writeCSV writes all results to a CSV file.
func writeCSV(path string, results []cellResult, commit ...string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	if len(commit) > 0 && commit[0] != "" {
		if err := w.Write([]string{"# commit", commit[0]}); err != nil {
			return err
		}
	}
	header := []string{
		"model", "kv_mode", "context_size", "epoch",
		"prefill_tokens", "prefill_tok_sec",
		"decode_tokens", "decode_tok_sec", "ppl",
		"kv_cache_mb_est", "size_vram_mb", "size_total_mb",
		"fully_offloaded", "block_count", "est_gpu_layers",
		"fit_status", "error",
	}
	if err := w.Write(header); err != nil {
		return err
	}

	for _, r := range results {
		pplStr := ""
		if r.PPL > 0 {
			pplStr = fmt.Sprintf("%.4f", r.PPL)
		}
		row := []string{
			r.Model,
			r.KVMode,
			strconv.Itoa(r.ContextSize),
			strconv.Itoa(r.Epoch),
			strconv.Itoa(r.PrefillTokens),
			fmt.Sprintf("%.2f", r.PrefillTokSec),
			strconv.Itoa(r.DecodeTokens),
			fmt.Sprintf("%.2f", r.DecodeTokSec),
			pplStr,
			fmt.Sprintf("%.1f", r.KVCacheMB),
			fmt.Sprintf("%.1f", r.SizeVRAMMB),
			fmt.Sprintf("%.1f", r.SizeTotalMB),
			strconv.FormatBool(r.FullyOffloaded),
			strconv.Itoa(r.BlockCount),
			strconv.Itoa(r.EstGPULayers),
			r.FitStatus,
			r.Error,
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return nil
}

// avgKey is the grouping key for summary averaging.
type avgKey struct {
	Model   string
	Ctx     int
	KVMode  string
}

// avgVal accumulates sums for averaging.
type avgVal struct {
	PrefillSum   float64
	DecodeSum    float64
	PPLSum       float64
	PPLCount     int // separate count: PPL may be 0 for warmup runs
	KVCacheMB    float64
	SizeVRAMMB   float64
	SizeTotalMB  float64
	EstGPULayers int
	FitStatus    string
	Count        int
}

// writeSummaryMarkdown writes a Markdown summary of the benchmark results.
func writeSummaryMarkdown(w io.Writer, results []cellResult, kvModes []string, commit string) {
	fmt.Fprintf(w, "# TurboQuant A/B Benchmark Results\n\n")
	fmt.Fprintf(w, "Generated: %s\n\n", time.Now().Format(time.RFC3339))
	if commit != "" {
		fmt.Fprintf(w, "Commit: `%s`\n\n", commit)
	}

	// Deterministic KV mode ordering (fallback to alphabetical for unknowns)
	kvRank := map[string]int{"f16": 0, "tq4": 1, "tq4k": 2, "tq3": 3, "tq3k": 4, "tq2": 5, "tq2k": 6, "q8_0": 7, "q4_0": 8, "f32": 9}
	sortKey := func(mode string) int {
		if r, ok := kvRank[mode]; ok { return r }
		return len(kvRank)
	}

	type agg struct {
		pref, dec, vram, ppl float64
		pplCnt			   int
		fit				  string
		cnt				  int
	}

	// data[model][ctx][kvMode] = *agg
	data := make(map[string]map[int]map[string]*agg)
	var models []string
	var ctxs []int
	seenM, seenC := make(map[string]bool), make(map[int]bool)
	var anomalies []string

	for _, r := range results {
		if r.FitStatus == "oom" || r.FitStatus == "error" {
			anomalies = append(anomalies, fmt.Sprintf("[%s] Ctx %d %s: %s", r.Model, r.ContextSize, r.KVMode, r.Error))
			continue
		}
		if !seenM[r.Model] { seenM[r.Model] = true; models = append(models, r.Model) }
		if !seenC[r.ContextSize] { seenC[r.ContextSize] = true; ctxs = append(ctxs, r.ContextSize) }

		if data[r.Model] == nil { data[r.Model] = make(map[int]map[string]*agg) }
		if data[r.Model][r.ContextSize] == nil { data[r.Model][r.ContextSize] = make(map[string]*agg) }

		v := data[r.Model][r.ContextSize][r.KVMode]
		if v == nil {
			v = &agg{}
			data[r.Model][r.ContextSize][r.KVMode] = v
		}
		v.pref += r.PrefillTokSec
		v.dec += r.DecodeTokSec
		v.vram += r.SizeVRAMMB
		if r.PPL > 0 {
			v.ppl += r.PPL
			v.pplCnt++
			anomalies = append(anomalies, fmt.Sprintf("[%s] Ctx %d %s: PPL=%.4f (expected -)", r.Model, r.ContextSize, r.KVMode))
		}
		v.fit = r.FitStatus
		if r.FitStatus != "ok" {
			anomalies = append(anomalies, fmt.Sprintf("[%s] Ctx %d %s: Status=%s (expected ok)", r.Model, r.ContextSize, r.KVMode, r.FitStatus))
		}
		v.cnt++
	}
	sort.Ints(ctxs)

	// Unique KV modes present in valid results
	kvSet := make(map[string]bool)
	for _, r := range results {
		if r.FitStatus != "oom" && r.FitStatus != "error" {
			kvSet[r.KVMode] = true
		}
	}
	var modes []string
	for k := range kvSet { modes = append(modes, k) }
	sort.Slice(modes, func(i, j int) bool {
		ri, rj := sortKey(modes[i]), sortKey(modes[j])
		return ri < rj || (ri == rj && modes[i] < modes[j])
	})

	baseline := "f16"
	if !kvSet[baseline] && len(modes) > 0 {
		baseline = modes[0]
	}

	// Generic table generator
	genTable := func(title string, val func(*agg) float64) {
		fmt.Fprintf(w, "# %s\n\n", title)
		hdr := []string{"Model", "Ctx"}
		for _, m := range modes { hdr = append(hdr, m) }
		fmt.Fprintf(w, "| %s |\n", strings.Join(hdr, " | "))
		fmt.Fprintf(w, "| %s \n", strings.Repeat("-----|", len(hdr)))

		for _, mdl := range models {
			for _, ctx := range ctxs {
				row := []string{mdl, strconv.Itoa(ctx)}
				var baseVal float64
				var hasBase bool
				if bd, ok := data[mdl][ctx][baseline]; ok && bd.cnt > 0 {
					baseVal = val(bd) / float64(bd.cnt)
					hasBase = true
				}

				for _, m := range modes {
					v := data[mdl][ctx][m]
					if v == nil || v.cnt == 0 {
						row = append(row, "-")
						continue
					}
					avg := val(v) / float64(v.cnt)
					suffix := " tok/s"

					if title == "VRAM" {
						suffix = " MB"
					}

					if m == baseline || !hasBase || baseVal == 0 {
						row = append(row, fmt.Sprintf("%.1f%s", avg, suffix))
					} else {
						percent := ((avg - baseVal) * 100 / baseVal)
						if title == "VRAM" {
							if percent < avg {
								percent = -percent
							}
						} else {
							if percent > avg {
								percent = -percent
							}
						}
						row = append(row, fmt.Sprintf("%+.1f%% (%.0f%s)", percent, avg, suffix))
					}
				}
				fmt.Fprintf(w, "| %s |\n", strings.Join(row, " | "))
			}
		}
		fmt.Fprintln(w)
	}

	genTable("VRAM", func(a *agg) float64 { return a.vram })
	genTable("Prefill tok/s", func(a *agg) float64 { return a.pref })
	genTable("Decode tok/s", func(a *agg) float64 { return a.dec })

	if len(anomalies) > 0 {
		fmt.Fprintln(w, "## Notes on Non-Standard Results\n")
		seen := make(map[string]bool)
		for _, a := range anomalies {
			if !seen[a] {
				seen[a] = true
				fmt.Fprintf(w, "- %s\n", a)
			}
		}
		fmt.Fprintln(w)
	}
}

// tokSec converts token count and duration to tokens/second. Avoids div-by-zero.
func tokSec(count int, dur time.Duration) float64 {
	if dur <= 0 || count <= 0 {
		return 0
	}
	return float64(count) / dur.Seconds()
}

// run executes the full benchmark matrix.
func run(cfg config) error {
	if err := os.MkdirAll(cfg.outputDir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", cfg.outputDir, err)
	}

	var results []cellResult
	total := len(cfg.models) * len(cfg.contexts) * len(cfg.kvModes)
	cellIdx := 0

	for _, model := range cfg.models {
		for _, ctxSize := range cfg.contexts {
			for _, kvMode := range cfg.kvModes {
				cellIdx++
				fmt.Fprintf(os.Stderr, "[%d/%d] model=%s ctx=%d kv=%s\n",
					cellIdx, total, model, ctxSize, kvMode)

				// Start server for this KV mode
				serverCmd, err := startServer(cfg.binary, kvMode, cfg.port, cfg.cudaDevice, cfg.modelsDir, cfg.debug, cfg.tqFusedOff, cfg.flashAttn)
				if err != nil {
					fmt.Fprintf(os.Stderr, "ERROR: startServer: %v\n", err)
					appendError(&results, model, kvMode, ctxSize, "error", err.Error())
					continue
				}

				if err := waitForHealth(cfg.port, 30*time.Second); err != nil {
					fmt.Fprintf(os.Stderr, "ERROR: waitForHealth: %v\n", err)
					stopServer(serverCmd)
					appendError(&results, model, kvMode, ctxSize, "error", err.Error())
					continue
				}

				client, err := makeClient(cfg.port)
				if err != nil {
					fmt.Fprintf(os.Stderr, "ERROR: makeClient: %v\n", err)
					stopServer(serverCmd)
					appendError(&results, model, kvMode, ctxSize, "error", err.Error())
					continue
				}

				// Fetch model architecture info
				infoCtx, infoCancel := context.WithTimeout(context.Background(), 15*time.Second)
				mInfo := fetchModelInfo(infoCtx, client, model)
				infoCancel()

				kvCacheMB := estimateKVCacheMB(mInfo, ctxSize, kvMode)

				// Warmup phase (negative epoch indices)
				oomDuringWarmup := false
				for i := 0; i < cfg.warmup; i++ {
					_, _, _, _, _, wErr := benchmarkCell(
						client, model, ctxSize, cfg.predict, cfg.maxPromptTokens, -(i + 1), cfg.timeout, cfg.debug, false,
					)
					if wErr != nil {
						fmt.Fprintf(os.Stderr, "WARNING: warmup %d/%d failed: %v\n", i+1, cfg.warmup, wErr)
						if isOOM(wErr) {
							oomDuringWarmup = true
							break
						}
					} else if cfg.debug {
						fmt.Fprintf(os.Stderr, "warmup %d/%d complete\n", i+1, cfg.warmup)
					}
				}

				if oomDuringWarmup {
					appendError(&results, model, kvMode, ctxSize, "oom", "OOM during warmup")
					unloadModel(client, model)
					time.Sleep(1 * time.Second)
					stopServer(serverCmd)
					time.Sleep(2 * time.Second)
					continue
				}

				// VRAM after warmup
				vramCtx, vramCancel := context.WithTimeout(context.Background(), 10*time.Second)
				sizeVRAM, sizeTotal := fetchVRAM(vramCtx, client, model)
				vramCancel()

				sizeVRAMMB := float64(sizeVRAM) / (1024 * 1024)
				sizeTotalMB := float64(sizeTotal) / (1024 * 1024)
				fullyOffloaded := sizeTotal > 0 && sizeVRAM >= int64(float64(sizeTotal)*0.95)

				estGPULayers := 0
				if sizeTotal > 0 {
					estGPULayers = int(math.Round(float64(mInfo.BlockCount) * float64(sizeVRAM) / float64(sizeTotal)))
				}

				fitStatus := "ok"
				if !fullyOffloaded && sizeVRAM > 0 {
					fitStatus = "partial_offload"
				}

				// Timed epochs
				for epoch := 0; epoch < cfg.epochs; epoch++ {
					prefillTok, prefillDur, decodeTok, decodeDur, cellPPL, eErr := benchmarkCell(
						client, model, ctxSize, cfg.predict, cfg.maxPromptTokens, epoch, cfg.timeout, cfg.debug, cfg.ppl,
					)

					r := cellResult{
						Model:          model,
						KVMode:         kvMode,
						ContextSize:    ctxSize,
						Epoch:          epoch,
						KVCacheMB:      kvCacheMB,
						SizeVRAMMB:     sizeVRAMMB,
						SizeTotalMB:    sizeTotalMB,
						FullyOffloaded: fullyOffloaded,
						BlockCount:     mInfo.BlockCount,
						EstGPULayers:   estGPULayers,
						FitStatus:      fitStatus,
					}

					if eErr != nil {
						r.FitStatus = "error"
						if isOOM(eErr) {
							r.FitStatus = "oom"
						}
						r.Error = eErr.Error()
						fmt.Fprintf(os.Stderr, "  epoch %d ERROR: %v\n", epoch, eErr)
					} else {
						r.PrefillTokens = prefillTok
						r.PrefillTokSec = tokSec(prefillTok, prefillDur)
						r.DecodeTokens = decodeTok
						r.DecodeTokSec = tokSec(decodeTok, decodeDur)
						r.PPL = cellPPL
						if cfg.ppl && cellPPL > 0 {
							fmt.Fprintf(os.Stderr, "  epoch %d: prefill=%.1f tok/s decode=%.1f tok/s PPL=%.4f VRAM=%.4f MB\n",
								epoch, r.PrefillTokSec, r.DecodeTokSec, r.PPL, r.SizeVRAMMB)
						} else {
							fmt.Fprintf(os.Stderr, "  epoch %d: prefill=%.1f tok/s decode=%.1f tok/s VRAM=%.4f MB\n",
								epoch, r.PrefillTokSec, r.DecodeTokSec, r.SizeVRAMMB)
						}
					}
					results = append(results, r)
				}

				unloadModel(client, model)
				time.Sleep(2 * time.Second)
				stopServer(serverCmd)
				time.Sleep(8 * time.Second) // allow CUDA to fully release VRAM before next cell
			}
		}
	}

	// Write outputs
	ts := time.Now().Format("20060102-150405")
	slug := ts
	if cfg.commit != "" {
		slug = fmt.Sprintf("%s-%s", ts, cfg.commit)
	}
	csvPath := filepath.Join(cfg.outputDir, fmt.Sprintf("tqbench-%s.csv", slug))
	if err := writeCSV(csvPath, results, cfg.commit); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: writeCSV: %v\n", err)
	} else {
		fmt.Fprintf(os.Stderr, "CSV written to %s\n", csvPath)
	}

	mdPath := filepath.Join(cfg.outputDir, fmt.Sprintf("tqbench-%s.md", slug))
	mdFile, err := os.Create(mdPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: create md: %v\n", err)
	} else {
		defer mdFile.Close()
		mw := io.MultiWriter(mdFile, os.Stderr)
		writeSummaryMarkdown(mw, results, cfg.kvModes, cfg.commit)
		fmt.Fprintf(os.Stderr, "Markdown written to %s\n", mdPath)
	}

	return nil
}

// appendError appends a single error result for a (model, kvMode, contextSize) cell.
func appendError(results *[]cellResult, model, kvMode string, ctxSize int, fitStatus, errMsg string) {
	*results = append(*results, cellResult{
		Model:     model,
		KVMode:    kvMode,
		ContextSize: ctxSize,
		FitStatus: fitStatus,
		Error:     errMsg,
	})
}

// isOOM heuristically detects out-of-memory errors.
func isOOM(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "oom") ||
		strings.Contains(msg, "out of memory") ||
		strings.Contains(msg, "cuda error") ||
		strings.Contains(msg, "failed to allocate")
}

func main() {
	var (
		binary     = flag.String("binary", "ollama", "Path to ollama binary")
		modelsStr  = flag.String("models", "llama3.2:3b,llama3.1:8b,qwen2.5:7b", "Comma-separated list of models")
		contextsStr = flag.String("contexts", "4096,32768,65536", "Comma-separated context sizes")
		kvModesStr  = flag.String("kv-modes", "f16,tq2,tq3,tq4,tq4k", "Comma-separated KV cache modes")
		epochs     = flag.Int("epochs", 3, "Number of timed epochs per cell")
		warmup     = flag.Int("warmup", 1, "Number of warmup iterations per cell")
		predict    = flag.Int("predict", 128, "Number of decode tokens to generate")
		maxPromptTokens = flag.Int("max-prompt-tokens", 0, "Cap on prompt length in tokens (0=contextSize-predict-64, capped at 16384). Set to a small value like 512 to speed up large-context runs while still allocating full KV cache.")
		port       = flag.Int("port", 11434, "Port for ollama server")
		outputDir  = flag.String("output", "./results", "Directory for output files")
		timeoutStr = flag.String("timeout", "10m", "Per-request timeout (e.g. 10m, 5m30s)")
		cudaDev    = flag.String("cuda-device", "", "CUDA_VISIBLE_DEVICES override (default: let ollama auto-detect)")
		debug      = flag.Bool("debug", false, "Enable debug output (server stderr, response tokens)")
		modelsDir  = flag.String("models-dir", "", "OLLAMA_MODELS directory (overrides default ~/.ollama/models)")
		tqFused    = flag.Bool("tq-fused-off", false, "Deprecated: retained for CLI compatibility; fused-kernel gating was removed")
		flashAttn  = flag.Bool("fa", true, "Enable Flash Attention in the spawned ollama server (OLLAMA_FLASH_ATTENTION=1/0)")
		ppl        = flag.Bool("ppl", false, "Collect per-token logprobs and compute decode perplexity (adds slight overhead)")
		commit     = flag.String("commit", "", "Git commit hash to stamp in output filenames and markdown header")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "TurboQuant A/B benchmark — compares KV cache modes across models and context sizes.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	// Parse timeout
	timeout, err := time.ParseDuration(*timeoutStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: invalid timeout %q: %v\n", *timeoutStr, err)
		os.Exit(1)
	}

	// Parse models
	models := strings.Split(*modelsStr, ",")
	for i, m := range models {
		models[i] = strings.TrimSpace(m)
	}

	// Parse contexts
	var contexts []int
	for _, s := range strings.Split(*contextsStr, ",") {
		s = strings.TrimSpace(s)
		n, err := strconv.Atoi(s)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: invalid context size %q: %v\n", s, err)
			os.Exit(1)
		}
		contexts = append(contexts, n)
	}

	// Parse KV modes
	kvModes := strings.Split(*kvModesStr, ",")
	for i, m := range kvModes {
		kvModes[i] = strings.TrimSpace(m)
	}

	cfg := config{
		binary:          *binary,
		models:          models,
		contexts:        contexts,
		kvModes:         kvModes,
		epochs:          *epochs,
		warmup:          *warmup,
		predict:         *predict,
		maxPromptTokens: *maxPromptTokens,
		port:            *port,
		outputDir:       *outputDir,
		timeout:         timeout,
		cudaDevice:      *cudaDev,
		modelsDir:       *modelsDir,
		debug:           *debug,
		tqFusedOff:      *tqFused,
		flashAttn:       *flashAttn,
		ppl:             *ppl,
		commit:          *commit,
	}

	if err := run(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "FATAL: %v\n", err)
	}

	// Just stop doing anything at the end
	for {
		time.Sleep(time.Hour)
	}
}
