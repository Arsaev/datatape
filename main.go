package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

type Config struct {
	SampleRate       int
	SamplesPerSymbol int
	Frequencies      []float64 // len = bands, bands must be power of two
	BitsPerSymbol    int
	SyncSeconds      float64 // lead-in pure tone seconds
	PreambleSymbols  int     // alternating low/high symbols for alignment
	Amplitude        float64 // peak amplitude before clipping (0..0.95)
}

func defaultFreqs(bands int) []float64 {
	switch bands {
	case 4:
		return []float64{500, 1500, 3000, 6000}
	case 8:
		return []float64{400, 800, 1500, 2500, 3500, 4500, 5500, 6500}
	case 6:
		return []float64{400, 900, 1600, 2600, 3800, 5200}
	default:
		// reasonable spread 300–7000 Hz (logish)
		f := make([]float64, bands)
		minF, maxF := 300.0, 7000.0
		for i := 0; i < bands; i++ {
			r := float64(i) / float64(bands-1)
			// log spacing
			f[i] = minF * math.Exp(r*math.Log(maxF/minF))
		}
		return f
	}
}

func bitsPerSymbol(nBands int) (int, error) {
	bps := 0
	for (1 << bps) < nBands {
		bps++
	}
	if 1<<bps != nBands {
		return 0, fmt.Errorf("bands must be power of two; got %d", nBands)
	}
	return bps, nil
}

func hannWindow(n int) []float64 {
	w := make([]float64, n)
	if n == 1 {
		w[0] = 1
		return w
	}
	for i := 0; i < n; i++ {
		w[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1)))
	}
	return w
}

func makeConfig(sr, sps, bands int, freqsCSV string) (Config, error) {
	if sr <= 0 || sps <= 0 || bands <= 1 {
		return Config{}, errors.New("invalid sr/sps/bands")
	}
	var freqs []float64
	if strings.TrimSpace(freqsCSV) != "" {
		parts := strings.Split(freqsCSV, ",")
		if len(parts) != bands {
			return Config{}, fmt.Errorf("expected %d freqs, got %d", bands, len(parts))
		}
		freqs = make([]float64, bands)
		for i, p := range parts {
			v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
			if err != nil || v <= 0 {
				return Config{}, fmt.Errorf("bad freq %q", p)
			}
			freqs[i] = v
		}
	} else {
		freqs = defaultFreqs(bands)
	}
	bps, err := bitsPerSymbol(bands)
	if err != nil {
		return Config{}, err
	}
	return Config{
		SampleRate:       sr,
		SamplesPerSymbol: sps,
		Frequencies:      freqs,
		BitsPerSymbol:    bps,
		SyncSeconds:      0.5,
		PreambleSymbols:  32,
		Amplitude:        0.9,
	}, nil
}

// writeWAV16 writes a mono PCM WAV file with 16-bit samples.
func writeWAV16(path string, sr int, samples []int16) error {
	var buf bytes.Buffer
	// RIFF header
	buf.WriteString("RIFF")
	binary.Write(&buf, binary.LittleEndian, uint32(36+len(samples)*2))
	buf.WriteString("WAVE")

	// fmt chunk
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, uint32(16))   // PCM fmt chunk size
	binary.Write(&buf, binary.LittleEndian, uint16(1))    // PCM
	binary.Write(&buf, binary.LittleEndian, uint16(1))    // mono
	binary.Write(&buf, binary.LittleEndian, uint32(sr))   // sample rate
	binary.Write(&buf, binary.LittleEndian, uint32(sr*2)) // byte rate
	binary.Write(&buf, binary.LittleEndian, uint16(2))    // block align
	binary.Write(&buf, binary.LittleEndian, uint16(16))   // bits per sample
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, uint32(len(samples)*2)) // data size
	for _, s := range samples {
		binary.Write(&buf, binary.LittleEndian, s)
	}
	return os.WriteFile(path, buf.Bytes(), 0644)
}

func readWAV16Mono(path string) (sr int, samples []int16, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, nil, err
	}
	r := bytes.NewReader(b)
	var riff [4]byte
	if _, err = io.ReadFull(r, riff[:]); err != nil {
		return
	}
	if string(riff[:]) != "RIFF" {
		err = errors.New("not RIFF")
		return
	}
	var fileSize uint32
	binary.Read(r, binary.LittleEndian, &fileSize)
	var wave [4]byte
	io.ReadFull(r, wave[:])
	if string(wave[:]) != "WAVE" {
		err = errors.New("not WAVE")
		return
	}
	var fmtFound, dataFound bool
	var channels, bits uint16
	var byteRate uint32
	for {
		var id [4]byte
		if _, err = io.ReadFull(r, id[:]); err != nil {
			break
		}
		var sz uint32
		if err = binary.Read(r, binary.LittleEndian, &sz); err != nil {
			return
		}
		switch string(id[:]) {
		case "fmt ":
			var audioFmt uint16
			binary.Read(r, binary.LittleEndian, &audioFmt)
			binary.Read(r, binary.LittleEndian, &channels)
			var srate uint32
			binary.Read(r, binary.LittleEndian, &srate)
			binary.Read(r, binary.LittleEndian, &byteRate)
			var blockAlign uint16
			binary.Read(r, binary.LittleEndian, &blockAlign)
			binary.Read(r, binary.LittleEndian, &bits)
			if sz > 16 {
				r.Seek(int64(sz-16), io.SeekCurrent)
			}
			if audioFmt != 1 || channels != 1 || bits != 16 {
				return 0, nil, fmt.Errorf("need PCM16 mono; got fmt=%d ch=%d bits=%d", audioFmt, channels, bits)
			}
			sr = int(srate)
			fmtFound = true
		case "data":
			if !fmtFound {
				return 0, nil, errors.New("fmt before data expected")
			}
			n := int(sz / 2)
			samples = make([]int16, n)
			for i := 0; i < n; i++ {
				binary.Read(r, binary.LittleEndian, &samples[i])
			}
			dataFound = true
		default:
			r.Seek(int64(sz), io.SeekCurrent)
		}
		if fmtFound && dataFound {
			break
		}
	}
	if !dataFound {
		return 0, nil, errors.New("no data chunk")
	}
	return sr, samples, nil
}

// synthTone generates a sine wave tone with given frequency, duration, sample rate, amplitude, and phase.
// butterworthLowPass applies a simple 2nd-order Butterworth low-pass filter to mono PCM samples.
func butterworthLowPass(samples []int16, sr int, cutoff float64) []int16 {
	// Coefficients for 2nd-order Butterworth
	// Reference: https://www.dsprelated.com/showarticle/1119.php
	wc := 2 * math.Pi * cutoff / float64(sr)
	k := math.Tan(wc / 2)
	norm := 1 / (1 + math.Sqrt(2)*k + k*k)
	a0 := k * k * norm
	a1 := 2 * a0
	a2 := a0
	b1 := 2 * (k*k - 1) * norm
	b2 := (1 - math.Sqrt(2)*k + k*k) * norm

	// Filter state
	var x1, x2, y1, y2 float64
	out := make([]int16, len(samples))
	for i := 0; i < len(samples); i++ {
		x0 := float64(samples[i])
		y0 := a0*x0 + a1*x1 + a2*x2 - b1*y1 - b2*y2
		// Clamp to int16
		if y0 > 32767 {
			y0 = 32767
		}
		if y0 < -32768 {
			y0 = -32768
		}
		out[i] = int16(y0)
		x2 = x1
		x1 = x0
		y2 = y1
		y1 = y0
	}
	return out
}
func synthTone(freq float64, n int, sr int, amp float64, phaseStart float64, win []float64) ([]float64, float64) {
	y := make([]float64, n)
	phase := phaseStart
	dp := 2 * math.Pi * freq / float64(sr)
	for i := 0; i < n; i++ {
		s := math.Sin(phase) * amp
		if win != nil {
			s *= win[i]
		}
		y[i] = s
		phase += dp
	}
	// keep phase continuous to reduce clicks
	return y, math.Mod(phase, 2*math.Pi)
}

func packBitsToSymbols(data []byte, bitsPerSym int) []int {
	mask := (1 << bitsPerSym) - 1
	var symbols []int
	bitbuf := 0
	bitcnt := 0
	for _, b := range data {
		bitbuf |= int(b) << bitcnt
		bitcnt += 8
		for bitcnt >= bitsPerSym {
			s := bitbuf & mask
			symbols = append(symbols, s)
			bitbuf >>= bitsPerSym
			bitcnt -= bitsPerSym
		}
	}
	if bitcnt > 0 {
		symbols = append(symbols, bitbuf&mask) // pad with zeros
	}
	return symbols
}

func symbolsToBits(symbols []int, bitsPerSym int) []byte {
	out := make([]byte, 0, len(symbols)*bitsPerSym/8+8)
	acc := 0
	nacc := 0
	for _, s := range symbols {
		acc |= (s & ((1 << bitsPerSym) - 1)) << nacc
		nacc += bitsPerSym
		for nacc >= 8 {
			out = append(out, byte(acc&0xFF))
			acc >>= 8
			nacc -= 8
		}
	}
	if nacc > 0 {
		out = append(out, byte(acc&0xFF))
	}
	return out
}

func encodeFileToWAV(inPath, outPath string, cfg Config) error {
	data, err := os.ReadFile(inPath)
	if err != nil {
		return err
	}
	// Error correction: block parity
	blockSize := 16
	var eccBuf bytes.Buffer
	binary.Write(&eccBuf, binary.LittleEndian, uint32(len(data)))
	for i := 0; i < len(data); i += blockSize {
		end := i + blockSize
		if end > len(data) {
			end = len(data)
		}
		block := data[i:end]
		parity := byte(0)
		for _, b := range block {
			parity ^= b
		}
		eccBuf.Write(block)
		eccBuf.WriteByte(parity)
	}
	payload := eccBuf.Bytes()

	symbols := packBitsToSymbols(payload, cfg.BitsPerSymbol)

	// Build waveform: silence, sync tone, preamble, payload
	silenceN := int(0.25 * float64(cfg.SampleRate))
	syncN := int(cfg.SyncSeconds * float64(cfg.SampleRate))
	sps := cfg.SamplesPerSymbol
	win := hannWindow(sps)
	amp := cfg.Amplitude

	total := make([]float64, 0, silenceN+syncN+len(symbols)*sps+cfg.PreambleSymbols*sps*2)
	// 0.25 s silence
	total = append(total, make([]float64, silenceN)...)

	// Sync tone at highest frequency
	phase := 0.0
	syncWave, phase := synthTone(cfg.Frequencies[len(cfg.Frequencies)-1], syncN, cfg.SampleRate, amp*0.8, phase, nil)
	total = append(total, syncWave...)

	// Preamble: alternate low/high indexes
	for i := 0; i < cfg.PreambleSymbols; i++ {
		idx := 0
		if i%2 == 1 {
			idx = len(cfg.Frequencies) - 1
		}
		w, p := synthTone(cfg.Frequencies[idx], sps, cfg.SampleRate, amp, phase, win)
		phase = p
		total = append(total, w...)
	}

	// Payload symbols
	for _, s := range symbols {
		f := cfg.Frequencies[s]
		w, p := synthTone(f, sps, cfg.SampleRate, amp, phase, win)
		phase = p
		total = append(total, w...)
	}

	// Normalize to int16
	var maxAbs float64
	for _, v := range total {
		if a := math.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}
	norm := 1.0
	if maxAbs > 0.99 {
		norm = 0.99 / maxAbs
	}
	ints := make([]int16, len(total))
	for i, v := range total {
		x := v * norm
		if x > 1 {
			x = 1
		}
		if x < -1 {
			x = -1
		}
		ints[i] = int16(math.Round(x * 32767))
	}

	return writeWAV16(outPath, cfg.SampleRate, ints)
}

// goertzelMag computes the magnitude of a frequency using the Goertzel algorithm.
func goertzelMag(samples []float64, freq float64, sr int) float64 {
	n := len(samples)
	k := 0.5 + float64(n)*freq/float64(sr)
	w := 2 * math.Cos(2*math.Pi*k/float64(n))

	s1, s2 := 0.0, 0.0
	for _, x := range samples {
		s0 := x + w*s1 - s2
		s2, s1 = s1, s0
	}

	// Magnitude squared
	mag2 := s1*s1 + s2*s2 - w*s1*s2
	return math.Sqrt(mag2)
}

func findSyncStart(samples []int16, sr int, syncFreq float64, minDuration float64) int {
	minSamples := int(minDuration * float64(sr))
	windowSize := sr / 20 // 0.05 second windows for better precision
	threshold := 0.2      // Lower threshold for better detection

	for i := 0; i <= len(samples)-minSamples; i += windowSize / 8 {
		end := i + windowSize
		if end > len(samples) {
			end = len(samples)
		}

		// Convert to float64 for Goertzel
		window := make([]float64, end-i)
		for j := i; j < end; j++ {
			window[j-i] = float64(samples[j]) / 32768.0
		}

		mag := goertzelMag(window, syncFreq, sr)
		if mag > threshold {
			// Found potential sync, verify sustained signal
			sustained := 0
			maxGap := windowSize / 4
			gapCount := 0

			for j := i; j < len(samples) && sustained < minSamples; j += windowSize / 8 {
				wEnd := j + windowSize
				if wEnd > len(samples) {
					break
				}

				wSamples := make([]float64, windowSize)
				for k := 0; k < windowSize && j+k < len(samples); k++ {
					wSamples[k] = float64(samples[j+k]) / 32768.0
				}

				if goertzelMag(wSamples, syncFreq, sr) > threshold {
					sustained += windowSize / 8
					gapCount = 0
				} else {
					gapCount++
					if gapCount > maxGap {
						break
					}
				}
			}

			if sustained >= minSamples {
				// Find more precise start by looking for signal rise
				for k := max(0, i-windowSize); k <= i+windowSize && k < len(samples)-windowSize; k++ {
					testWindow := make([]float64, windowSize/2)
					for m := 0; m < windowSize/2 && k+m < len(samples); m++ {
						testWindow[m] = float64(samples[k+m]) / 32768.0
					}
					if goertzelMag(testWindow, syncFreq, sr) > threshold*1.5 {
						return k
					}
				}
				return i
			}
		}
	}
	return -1
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func detectSymbol(samples []float64, freqs []float64, sr int) int {
	bestIdx := 0
	bestMag := 0.0
	var mags []float64

	for i := 0; i < len(freqs); i++ {
		mag := goertzelMag(samples, freqs[i], sr)
		mags = append(mags, mag)
		if mag > bestMag {
			bestMag = mag
			bestIdx = i
		}
	}

	// Confidence check make sure the best signal is significantly stronger
	secondBest := 0.0
	for i, mag := range mags {
		if i != bestIdx && mag > secondBest {
			secondBest = mag
		}
	}

	// If the signals are too close, might be noise/interference
	if bestMag > 0 && secondBest > 0 {
		ratio := bestMag / secondBest
		if ratio < 1.2 { // Less than 20% difference
			// Use some basic heuristic or previous symbol for context
			// For now, just proceed with best guess
		}
	}

	return bestIdx
}

// func skipPreamble(samples []int16, start int, cfg Config) int {
// 	sps := cfg.SamplesPerSymbol

// 	// Try to find the end of preamble by looking for pattern change
// 	// Preamble alternates between freq[0] and freq[max], so look for
// 	// when this pattern stops
// 	pos := start
// 	alternationCount := 0
// 	// expectedPattern := true

// 	for i := 0; i < cfg.PreambleSymbols && pos+sps <= len(samples); i++ {
// 		symSamples := make([]float64, sps)
// 		for j := 0; j < sps && pos+j < len(samples); j++ {
// 			symSamples[j] = float64(samples[pos+j]) / 32768.0
// 		}

// 		symbol := detectSymbol(symSamples, cfg.Frequencies, cfg.SampleRate)

// 		expectedSym := 0
// 		if i%2 == 1 {
// 			expectedSym = len(cfg.Frequencies) - 1
// 		}

// 		if symbol == expectedSym {
// 			alternationCount++
// 		}

// 		pos += sps
// 	}

// 	fmt.Printf("Preamble analysis: %d/%d symbols matched pattern\n", alternationCount, cfg.PreambleSymbols)

// 	// Return position after expected preamble length
// 	return start + cfg.PreambleSymbols*sps
// }

func decodeWAVToFile(inPath, outPath string, cfg Config) error {
	sr, samples, err := readWAV16Mono(inPath)
	if err != nil {
		return err
	}

	if sr != cfg.SampleRate {
		return fmt.Errorf("sample rate mismatch: expected %d, got %d", cfg.SampleRate, sr)
	}

	// Apply Butterworth low-pass filter to reduce tape noise
	maxFreq := cfg.Frequencies[len(cfg.Frequencies)-1]
	cutoff := maxFreq * 1.2 // 20% above max symbol freq
	samples = butterworthLowPass(samples, sr, cutoff)

	// Find sync signal
	syncFreq := cfg.Frequencies[len(cfg.Frequencies)-1]
	syncStart := findSyncStart(samples, sr, syncFreq, cfg.SyncSeconds*0.8)
	if syncStart < 0 {
		return errors.New("sync signal not found")
	}

	fmt.Printf("Sync found at sample %d (%.2fs)\n", syncStart, float64(syncStart)/float64(sr))

	// Skip sync + preamble with better alignment
	syncEndSample := syncStart + int(cfg.SyncSeconds*float64(sr))

	// Find actual preamble start by looking for alternating pattern
	preambleStart := syncEndSample
	sps := cfg.SamplesPerSymbol

	// Look ahead a bit to find the actual preamble
	searchRange := sps * 4 // Search within 4 symbol periods
	bestPreambleStart := preambleStart
	bestScore := 0

	for offset := 0; offset < searchRange && preambleStart+offset+sps*8 < len(samples); offset += sps / 4 {
		testStart := preambleStart + offset
		score := 0

		// Test first 8 symbols of preamble for alternating pattern
		for i := 0; i < 8 && testStart+i*sps+sps <= len(samples); i++ {
			symStart := testStart + i*sps
			symSamples := make([]float64, sps)
			for j := 0; j < sps; j++ {
				symSamples[j] = float64(samples[symStart+j]) / 32768.0
			}

			symbol := detectSymbol(symSamples, cfg.Frequencies, sr)
			expectedSym := 0
			if i%2 == 1 {
				expectedSym = len(cfg.Frequencies) - 1
			}

			if symbol == expectedSym {
				score++
			}
		}

		if score > bestScore {
			bestScore = score
			bestPreambleStart = testStart
		}
	}

	dataStart := bestPreambleStart + cfg.PreambleSymbols*sps

	fmt.Printf("Sync ends at sample %d, preamble starts at %d (score %d/8)\n",
		syncEndSample, bestPreambleStart, bestScore)

	if dataStart >= len(samples) {
		return errors.New("no data after preamble")
	}

	fmt.Printf("Data starts at sample %d (%.2fs)\n", dataStart, float64(dataStart)/float64(sr))

	// Decode symbols
	var symbols []int
	sps = cfg.SamplesPerSymbol

	for pos := dataStart; pos+sps <= len(samples); pos += sps {
		// Extract symbol samples and convert to float64
		symSamples := make([]float64, sps)
		for i := 0; i < sps; i++ {
			symSamples[i] = float64(samples[pos+i]) / 32768.0
		}

		symbol := detectSymbol(symSamples, cfg.Frequencies, sr)
		symbols = append(symbols, symbol)
	}

	fmt.Printf("Decoded %d symbols\n", len(symbols))

	// Convert symbols back to bytes
	bits := symbolsToBits(symbols, cfg.BitsPerSymbol)

	fmt.Printf("Converted to %d bytes\n", len(bits))

	if len(bits) < 4 {
		return errors.New("insufficient data for length header")
	}

	// Extract original length
	var origLen uint32
	buf := bytes.NewReader(bits[:4])
	binary.Read(buf, binary.LittleEndian, &origLen)

	fmt.Printf("Header indicates original length: %d bytes\n", origLen)
	fmt.Printf("Available data: %d bytes\n", len(bits)-4)

	// Sanity check on length
	if origLen > 1000000 { // 1MB limit
		return fmt.Errorf("unrealistic decoded length %d, likely corruption", origLen)
	}

	// Error correction: check parity blocks
	blockSize := 16
	dataWithParity := bits[4:]
	var recovered []byte
	for i := 0; i < len(dataWithParity); i += blockSize + 1 {
		end := i + blockSize
		if end > len(dataWithParity)-1 {
			end = len(dataWithParity) - 1
		}
		block := dataWithParity[i:end]
		if end >= len(dataWithParity) {
			break
		}
		parity := dataWithParity[end]
		calcParity := byte(0)
		for _, b := range block {
			calcParity ^= b
		}
		if parity != calcParity {
			fmt.Printf("Warning: parity error in block %d (offset %d)\n", i/(blockSize+1), i)
			// Could attempt single-bit correction here (not implemented)
		}
		recovered = append(recovered, block...)
	}
	if int(origLen) > len(recovered) {
		fmt.Printf("Warning: decoded length %d exceeds available data %d\n", origLen, len(recovered))
		origLen = uint32(len(recovered))
	}
	origData := recovered[:origLen]

	return os.WriteFile(outPath, origData, 0644)
}

func main() {
	var (
		encode   = flag.Bool("encode", false, "encode mode")
		decode   = flag.Bool("decode", false, "decode mode")
		input    = flag.String("input", "", "input file")
		output   = flag.String("output", "", "output file")
		sr       = flag.Int("sr", 44100, "sample rate")
		sps      = flag.Int("sps", 2205, "samples per symbol")
		bands    = flag.Int("bands", 4, "number of frequency bands (power of 2)")
		freqsCSV = flag.String("freqs", "", "comma-separated frequencies")
		showHelp = flag.Bool("help", false, "show help")
	)
	flag.Parse()

	if *showHelp || (!*encode && !*decode) || *input == "" || *output == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -encode|-decode -input FILE -output FILE [options]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nOptions:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  %s -encode -input data.txt -output audio.wav\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -decode -input audio.wav -output recovered.txt\n", os.Args[0])
		os.Exit(1)
	}

	cfg, err := makeConfig(*sr, *sps, *bands, *freqsCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Config: SR=%d, SPS=%d, Bands=%d, BPS=%d\n",
		cfg.SampleRate, cfg.SamplesPerSymbol, len(cfg.Frequencies), cfg.BitsPerSymbol)
	fmt.Printf("Frequencies: %v\n", cfg.Frequencies)

	if *encode {
		if err := encodeFileToWAV(*input, *output, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "Encode error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Encoded %s -> %s\n", *input, *output)
	} else {
		if err := decodeWAVToFile(*input, *output, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "Decode error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Decoded %s -> %s\n", *input, *output)
	}
}
