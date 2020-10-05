package bert

import (
	"fmt"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/model/wordpiece"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/processor"

	"github.com/sugarme/transformer/pretrained"
	"github.com/sugarme/transformer/util"
)

type Tokenizer struct {
	*tokenizer.Tokenizer
}

func NewTokenizer() *Tokenizer {
	tk := tokenizer.NewTokenizer(nil)
	return &Tokenizer{tk}
}

// Load method loads tokenizer from pretrained file.
//
// NOTE: for BERT model, only vocabNameOrPath will be used. `mergesNameOrPath` won't.
// It's there to satisfy the method Load signature of `pretrained.Tokenizer` interface.
func (bt *Tokenizer) Load(vocabNameOrPath, mergesNameOrPath string, params map[string]interface{}) error {
	var urlOrFilename string
	// If modelName, infer to default vocab filename:
	if vocabFile, ok := pretrained.BertVocabs[vocabNameOrPath]; ok {
		urlOrFilename = vocabFile
	} else {
		// Otherwise, just take the input
		urlOrFilename = vocabNameOrPath
	}

	cachedFile, err := util.CachedPath(urlOrFilename)
	if err != nil {
		return err
	}

	model, err := wordpiece.NewWordPieceFromFile(cachedFile, "[UNK]")
	if err != nil {
		return err
	}

	bt.WithModel(model)

	bertNormalizer := normalizer.NewBertNormalizer(true, true, true, true)
	bt.WithNormalizer(bertNormalizer)

	bertPreTokenizer := pretokenizer.NewBertPreTokenizer()
	bt.WithPreTokenizer(bertPreTokenizer)

	var specialTokens []tokenizer.AddedToken
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("[MASK]", true))

	bt.AddSpecialTokens(specialTokens)

	sepId, ok := bt.TokenToId("[SEP]")
	if !ok {
		return fmt.Errorf("Cannot find ID for [SEP] token.\n")
	}
	sep := processor.PostToken{Id: sepId, Value: "[SEP]"}

	clsId, ok := bt.TokenToId("[CLS]")
	if !ok {
		return fmt.Errorf("Cannot find ID for [CLS] token.\n")
	}
	cls := processor.PostToken{Id: clsId, Value: "[CLS]"}

	postProcess := processor.NewBertProcessing(sep, cls)
	bt.WithPostProcessor(postProcess)

	// TODO: update params

	return nil
}
