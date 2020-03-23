package pretokenizer

import (
	"fmt"
	"math"
	"regexp"
	"strings"

	"github.com/sugarme/sermo/normalizer"
	"github.com/sugarme/sermo/tokenizer"
	slice "github.com/sugarme/sermo/util/slice"
)

// Regular epxression to split string to `word` token
// including prefix whitespace. Contractions and punctuation
// will be split as well.
// Ref.https://regex101.com/r/pf5XJv
const splitRegStr = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`

var splitRE = regexp.MustCompile(splitRegStr)

var BytesChar map[uint8]string = GenerateBytesChar()

var CharBytes map[string]uint8 = func() map[string]uint8 {
	var bc = GenerateBytesChar()
	var cb map[string]uint8 = make(map[string]uint8)
	for b, c := range bc {
		cb[c] = b
	}
	return cb
}()

// BytesChar maps first 0-255 (byte) to first 0-255 `char` in unicode
// Ref. https://en.wikipedia.org/wiki/List_of_Unicode_characters
func GenerateBytesChar() map[uint8]string {
	var bs []uint8 // byte
	var bc map[uint8]string = make(map[uint8]string)

	// Basic latin
	for i := 33; i <= 126; i++ {
		bs = append(bs, uint8(i))
		bc[uint8(i)] = string(uint8(i))
	}

	// latin-1 supplement (excluding `173`)
	for i := 161; i <= 172 && i != 173; i++ {
		bs = append(bs, uint8(i))
		bc[uint8(i)] = string(i)
	}

	// Append `control` byte (0-32) and (127-160) and 173
	// Due to the control byte, first 256 runes will be shifted right 256
	var n = 0
	for i := 0; i <= 255; i++ {
		if !slice.Contain(uint8(i), bs) {
			// if !contain(uint8(i), bs) {
			bs = append(bs, uint8(i))
			bc[uint8(i)] = string(256 + n)
			n += 1
		}
	}

	return bc
}

// ByteLevel provides all the neccessary steps to handle the
// BPE tokenization at byte-level. It takes care of all the required
// processing steps to transform a utf-8 string as needed before and
// after the BPE model does it job.
type ByteLevel struct {
	// whether to add a leading space to the first word.
	// It allows to treat the leading word just as any other words.
	AddPrefixSpace bool

	// Whether the post processing step should trim offsets
	// to avoid including whitespaces.
	TrimOffsets bool
}

// NewByteLevel returns a default ByteLevel with both
// AddPrefixSpace and TrimOffsets set true
func NewByteLevel() *ByteLevel {
	return &ByteLevel{
		AddPrefixSpace: true,
		TrimOffsets:    true,
	}
}

// Alphabet returns set of first 256 unicode `char`
func (bl *ByteLevel) Alphabet() map[string]struct{} {
	var ab = make(map[string]struct{})
	for _, c := range BytesChar {
		ab[c] = struct{}{}
	}

	return ab
}

// SetAddPrefixSpace set `AddPrefixSpace` property
func (bl *ByteLevel) SetAddPrefixSpace(v bool) {
	bl.AddPrefixSpace = v
}

// SetTrimOffsets set `TrimOffsets` property
func (bl *ByteLevel) SetTrimOffsets(v bool) {
	bl.TrimOffsets = v
}

// Implement `PreTokenizer` methods for `ByteLevel`

type PreTokResult struct {
	Content string
	Offsets tokenizer.Offsets
}

// PreTokenize transforms all the unicode characters into
// their byte-level counterpart. It also splits the input
// according to the configured regex.
func (bl *ByteLevel) PreTokenize(normalized normalizer.Normalized) *PreTokResult {

	var res []PreTokResult
	var positions []tokenizer.Offsets
	normalizedString := normalized.GetNormalized()

	if bl.AddPrefixSpace && !strings.HasPrefix(normalizedString, " ") {
		normalizedString = fmt.Sprintf(" %v", normalizedString)
	}

	// positions holds slice of matches' loc
	// (which is 2 element slice loc[0] start - inclusive
	// and loc[1] end - exclusive)
	locs := splitRE.FindAllStringIndex(normalizedString, -1)
	chars := strings.Split(normalizedString, "")

	for _, loc := range locs {
		start := loc[0]
		end := loc[1]
		// if last `char` is a whitespace, followed by a non-whitespace
		// remove this whitespace
		last := chars[loc[1]-1] // -1 because of end exclusive

		var next string
		// Exclude last `char` of string
		if loc[1] < len(normalizedString) {
			next = chars[0]
		}

		if last == " " && next != " " {
			end -= 1
		}

		// If first `char` is not a whitespace but the previous one
		// was, add that whitespace
		first := chars[loc[0]]
		var prev string
		// Exclude the first `char` of string
		if loc[0] > 0 {
			prev = chars[start-1]
		}
		if first != " " && prev == " " {
			start -= 1
		}

		positions = append(positions, tokenizer.Offsets{Start: uint(start), End: uint(end)})

	}

	// setup goroutine to split string concurrently based on `positions`
	type Split struct {
		Char    string
		Changes uint8
	}

	// TODO: implement concurrently split
	var splits [][]Split
	for _, pos := range positions {
		tok := normalizedString[pos.Start:(pos.End + 1)] // +1 to include `End` position
		tokChars := strings.Split(tok, "")

		var i = 0
		var split []Split
		var changeMap []normalizer.ChangeMap
		for _, c := range tokChars {
			size := len(c) // number of bytes for current `char`
			bytes := []byte(normalizedString[i:(i + size)])
			i += size

			for _, b := range bytes {
				var change uint8 = 0
				if i > 0 {
					change = 1
				}

				split = append(split, Split{Char: BytesChar[b], Changes: change})
				changeMap = append(changeMap, normalizer.ChangeMap{RuneVal: BytesChar[b], Changes: int(change)})
			}

			// Update normalizedString
			normalized.Transform(changeMap, 0)

		}

		splits = append(splits, split)
	}

	// Collect splits and their offsets
	var totalLen = 0
	for _, split := range splits {
		var len = 0
		var chars []string
		for _, c := range split {
			chars = append(chars, c.Char)
			len += 1
		}
		totalLen += len
		tok := strings.Join(chars, "")
		offsets := tokenizer.Offsets{Start: uint(totalLen - len), End: uint(totalLen)}
		res = append(res, PreTokResult{tok, offsets})
	}

	return &PreTokResult{}
}

// Implement Decoder for `ByteLevel`

// Decode converts any byte-level characters to their unicode couterpart
// before merging everything back into a single string
func (bl *ByteLevel) Decode(tokens []string) string {
	s := strings.Join(tokens, "")
	chars := strings.Split(s, "")

	var bytes []byte

	for _, c := range chars {
		b := CharBytes[c]

		bytes = append(bytes, b)
	}

	return string(bytes)
}

// Implement PostProcessor for ByteLevel
func (bl *ByteLevel) AddedToken(isPair bool) uint {
	return 0
}

func (bl *ByteLevel) Process(encoding tokenizer.Encoding, addSpecialTokens bool, pairEncodingOpt ...tokenizer.Encoding) []tokenizer.Encoding {

	// TODO: implement
	var finalEncoding []tokenizer.Encoding

	enc := processOffsets(bl.TrimOffsets, encoding)

	finalEncoding = append(finalEncoding, enc)
	if pairEncodingOpt != nil {
		pairEnc := processOffsets(bl.TrimOffsets, pairEncodingOpt[0])
		finalEncoding = append(finalEncoding, pairEnc)
	}

	return finalEncoding
}

func processOffsets(isTrimOffsets bool, encoding tokenizer.Encoding) tokenizer.Encoding {

	if !isTrimOffsets {
		return encoding
	}

	type Modif struct {
		LeadingSpaces uint
		TrailingSpace uint
	}

	var modifs []Modif
	var newOffsets []tokenizer.Offsets

	toks := encoding.GetTokens()

	for _, tok := range toks {

		var leadingSpaces uint = 0
		chars := strings.Split(tok, "")
		for _, c := range chars {
			if c != " " {
				break
			}
			leadingSpaces += 1
		}

		var trailingSpaces uint = 0
		for i := len(chars) - 1; i >= 0; i-- {
			if chars[i] != " " {
				break
			}
			trailingSpaces += 1
		}

		if leadingSpaces > 0 || trailingSpaces > 0 {
			modifs = append(modifs, Modif{
				LeadingSpaces: leadingSpaces,
				TrailingSpace: trailingSpaces,
			})
		}
	}

	for i, m := range modifs {
		offsets := encoding.GetOffsets()[i]

		if m.LeadingSpaces > 0 {
			minVal := math.Min(float64(offsets.Start+m.LeadingSpaces), float64(offsets.End))
			offsets.Start = uint(minVal)
		}

		if m.TrailingSpace > 0 {
			maxVal := math.Max(float64(offsets.End-m.TrailingSpace), float64(offsets.Start))
			offsets.End = uint(maxVal)
		}

		newOffsets = append(newOffsets, offsets)
	}

	encoding.Offsets = newOffsets

	return encoding
}
