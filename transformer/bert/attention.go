package bert

import (
	"github.com/sugarme/sermo/transformer/common"
	"github.com/sugarme/sermo/util/nn"
)

// BertSelfAttention:
//===================

type BertSelfAttention struct {
	NumAttentionHeads int64
	AttentionHeadSize int64
	Dropout           *common.Dropout
	OutputAttentions  bool
	Query             *nn.Linear
	Key               *nn.Linear
	Value             *nn.Linear
}

// BertSelfOutput:
//================

type BertSelfOutput struct {
	Linear    *nn.Linear
	LayerNorm *nn.LayerNorm
	Droput    *common.Dropout
}

// BertAttention:
//===============

type BertAttention struct {
	self   *BertSelfAttention
	Output *BertSelfOutput
}
