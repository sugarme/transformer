package pretokenizer

import (
	slice "github.com/sugarme/sermo/util/slice"
)

// BytesChar maps first 0-255 (byte) to first 0-255 `char` in unicode
// Ref. https://en.wikipedia.org/wiki/List_of_Unicode_characters
func BytesChar() map[uint8]string {
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
