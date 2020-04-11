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
	HiddenSize                int
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

// BertModel defines base architecture for BERT models.
// `Task-specific` models can be built from this base model.
// `Embeddings`: for `token`, `position` and `segment` embeddings
// `Encoder`: is a vector of layers. Each layer compose of a `self-attention`,
// an `intermedate` (linear) and an output ( linear + layer norm) sub-layers.
// `Pooler`: linear layer applied to the first element of the sequence (`[MASK]` token)
// `IsDecoder`: whether model is used as a decoder. If set to `true`
// a casual mask will be applied to hide future positions that should be attended to.
type BertModel struct {
	Embeddings *BertEmbedding
	Encoder    *BertEncoder
	Pooler     bool
	IsDecoder  bool
}
