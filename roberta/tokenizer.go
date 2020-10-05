package roberta

import (
	// "fmt"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/model/bpe"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/processor"

	"github.com/sugarme/transformer/pretrained"
	"github.com/sugarme/transformer/util"
)

// Tokenizer holds data for Roberta tokenizer.
type Tokenizer struct {
	*tokenizer.Tokenizer
}

// NewTokenizer creates a new Roberta tokenizer.
func NewTokenizer() *Tokenizer {
	tk := tokenizer.NewTokenizer(nil)
	return &Tokenizer{tk}
}

// Load loads Roberta tokenizer from pretrain vocab and merges files.
func (t *Tokenizer) Load(vocabNameOrPath, mergesNameOrPath string, params map[string]interface{}) error {
	var urlOrFilenameVocab string
	// If modelName, infer to default vocab filename:
	if vocabFile, ok := pretrained.RobertaVocabs[vocabNameOrPath]; ok {
		urlOrFilenameVocab = vocabFile
	} else {
		// Otherwise, just take the input
		urlOrFilenameVocab = vocabNameOrPath
	}
	cachedFileVocab, err := util.CachedPath(urlOrFilenameVocab)
	if err != nil {
		return err
	}

	var urlOrFilenameMerges string
	// If modelName, infer to default vocab filename:
	if mergesFile, ok := pretrained.RobertaMerges[mergesNameOrPath]; ok {
		urlOrFilenameMerges = mergesFile
	} else {
		// Otherwise, just take the input
		urlOrFilenameMerges = mergesNameOrPath
	}
	cachedFileMerges, err := util.CachedPath(urlOrFilenameMerges)
	if err != nil {
		return err
	}

	model, err := bpe.NewBpeFromFiles(cachedFileVocab, cachedFileMerges)
	if err != nil {
		return err
	}

	t.WithModel(model)

	bertNormalizer := normalizer.NewBertNormalizer(true, true, true, true)
	t.WithNormalizer(bertNormalizer)

	blPreTokenizer := pretokenizer.NewByteLevel()
	// blPreTokenizer.SetAddPrefixSpace(false)
	t.WithPreTokenizer(blPreTokenizer)

	var specialTokens []tokenizer.AddedToken
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<s>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<pad>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("</s>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<unk>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<mask>", true))
	t.AddSpecialTokens(specialTokens)

	postProcess := processor.DefaultRobertaProcessing()
	t.WithPostProcessor(postProcess)

	return nil
}
