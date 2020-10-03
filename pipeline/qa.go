package pipeline

import (
	"github.com/sugarme/gotch"
)

// Question Answering pipeline
// Extractive question answering from a given question and context. By default, the dependencies for this
// model will be downloaded for a DistilBERT model finetuned on SQuAD (Stanford Question Answering Dataset).
// Customized DistilBERT models can be loaded by overwriting the resources in the configuration.
// The dependencies will be downloaded to the user's home directory, under ~/.cache/transformer/distilbert-qa

type QAInput struct {
	Question string
	Context  string
}

type QAExample struct {
	Question         string
	Context          string
	DocTokens        []string
	CharToWordOffset []int
}

type QAFeature struct {
	InputIds         []int
	AttentionMask    []int
	TokenToOriginMap map[int]int
	PMask            []int8
	ExampleIndex     int
}

// Answer is output for question answering.
type Answer struct {
	Score  float64
	Start  int // start position of answer span
	End    int // end position of answer span
	Answer string
}

func removeDuplicates(items []interface{}) []interface{} {
	keys := make(map[interface{}]bool)
	list := []interface{}{}
	for _, item := range items {
		if _, value := keys[item]; !value {
			keys[item] = true
			list = append(list, item)
		}
	}
	return list
}

func NewQAExample(question string, context string) *QAExample {

	docTokens, charToWordOffset := splitContext(context)

	return &QAExample{
		Question:         question,
		Context:          context,
		DocTokens:        docTokens,
		CharToWordOffset: charToWordOffset,
	}
}

func splitContext(context string) ([]string, []int) {
	var docTokens []string
	var charToWordOffset []int
	var currentWord []rune
	var previousWhiteSpace bool = false

	for _, char := range context {
		charToWordOffset = append(charToWordOffset, len([]byte(string(char))))
		if isWhiteSpace(char) {
			previousWhiteSpace = true
			if len(currentWord) > 0 {
				docTokens = append(docTokens, string(currentWord))
			}
		} else {
			if previousWhiteSpace {
				currentWord = nil
			}

			currentWord = append(currentWord, char)
			previousWhiteSpace = false
		}
	}

	// Last word
	if len(currentWord) > 0 {
		docTokens = append(docTokens, string(currentWord))
	}

	return docTokens, charToWordOffset
}

func isWhiteSpace(char rune) bool {
	if char == ' ' || char == '\t' || char == '\r' || char == '\n' || char == 0x202F {
		return true
	}
	return false
}

// QuestionAnsweringConfig holds configuration for question answering
type QuestionAnsweringConfig struct {
	Model     string // model name or path
	Config    string // config name or path
	Vocab     string // vocab name or path
	Merges    string // merge name or path
	Device    gotch.Device
	ModelType ModelType
	LowerCase bool
}
