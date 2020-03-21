package pretokenizer

import (
	"regexp"

	"github.com/sugarme/sermo/normalizer"
	"github.com/sugarme/sermo/tokenizer"
	slice "github.com/sugarme/sermo/util/slice"
)

const constractRegStr = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`

var constractRE = regexp.MustCompile(constractRegStr)

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

// PreTokenize is in charge of transforming all the unicode
// characters into their byte-level counterpart. It also splits
// the input according to the configured regex.
func (bl *ByteLevel) PreTokenize(normalized normalizer.NormalizedString) *PreTokResult {
	// TODO: implement detail

	return &PreTokResult{}
}
