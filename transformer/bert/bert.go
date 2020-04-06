package bert

import (
	"github.com/sugarme/sermo/transformer/common"
)

// BertConfig defines the BERT model architecture (i.e., number of layers,
// hidden layer size, label mapping...)
type BertConfig struct {
	HiddenAct                 common.Activation
	AttentionProbsDropoutProb float64
	HiddenDropoutProb         float64
	HiddenSize                int64
	InitializerRnage          float32
	IntermediateSize          int64
	MaxPositionEmbeddings     int64
	NumAttentionHeads         int64
	NumHiddenLayers           int64
	TypeVocabSize             int64
	VocabSize                 int64
	OutputAttentions          bool
	OutputHiddenStates        bool
	IsDecoder                 bool
	Id2Label                  map[int64]string
	Label2Id                  map[string]int64
	NumLabels                 int64
}
