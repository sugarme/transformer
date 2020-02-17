// Basic text preprocessing tasks are:
// 1. Remove HTML tags
// 2. Remove extra whitespaces
// 3. Convert accented characters to ASCII characters
// 4. Expand contractions
// 5. Remove special characters
// 6. Lowercase all texts
// 7. Convert number words to numeric form
// 8. Remove numbers
// 9. Remove stopwords
// 10. Lemmatization
package normalizer

import (
	"encoding/csv"
	"encoding/json"
	"log"
	"os"
	"regexp"
	"strings"
)

type DefaultNormalizer struct {
	Lower           bool // to lowercase
	ExtraWhitespace bool // remove extra-whitespaces
	Contraction     bool // expand contraction
}

type DefaultOption func(*DefaultNormalizer)

func WithContractionExpansion() DefaultOption {
	return func(o *DefaultNormalizer) {
		o.Contraction = true
	}

}

func (dn DefaultNormalizer) Normalize(txt string) string {

	if dn.Lower {
		txt = toLowercase(txt)
	}

	if dn.ExtraWhitespace {
		txt = removeExtraWhitespace(txt)
	}

	if dn.Contraction {
		txt = expandContraction(txt)
	}

	return txt

}

func NewDefaultNormalizer(opts ...DefaultOption) DefaultNormalizer {

	dn := DefaultNormalizer{
		Lower:           true,
		ExtraWhitespace: true,
		Contraction:     false,
	}

	for _, o := range opts {
		o(&dn)
	}

	return dn

}

func toLowercase(txt string) string {
	return strings.ToLower(txt)
}

func removeExtraWhitespace(txt string) string {
	space := regexp.MustCompile(`\s+`)
	return space.ReplaceAllString(txt, " ")

}

func expandContraction(txt string) string {
	var cMap map[string]string
	cMap, err := loadContractionMap()
	if err != nil {
		log.Fatal(err)
	}

	if k, ok := cMap[txt]; ok {
		return k
	}

	return txt
}

func loadContractionMap() (map[string]string, error) {
	const file = "contraction.csv"
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}

	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}

	type Contract struct {
		Contraction string
		Expansion   string
	}

	var cList []Contract

	for _, line := range lines {
		cList = append(cList, Contract{
			Contraction: line[0],
			Expansion:   line[1],
		})
	}

	cMap := map[string]string{}
	inrec, _ := json.Marshal(cList)
	json.Unmarshal(inrec, &cMap)

	return cMap, nil

}
