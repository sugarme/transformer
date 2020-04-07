package bert

import (
	"github.com/sugarme/sermo/util/nn"

	G "gorgonia.org/gorgonia"
	ts "gorgonia.org/tensor"
)

// BertEmbedding defines interface for BertModel or RoBertaModel
type BertEmbedding interface {
	ForwardT(inputIds, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (ts.Tensor, string)
}

// Dropout is a func type with signature of Gorgonia Dropout func
type Dropout func(x *G.Node, prob float64) (retVal *G.Node, err error)

type BertEmbeddings struct {
	WordEmbeddings      nn.Embedding
	PositionEmbeddings  nn.Embedding
	TokenTypeEmbeddings nn.Embedding
	LayerNorm           nn.LayerNorm
	Dropout             Dropout
}

// NewBertEmbedding builds a new BertEmbedding
// * `p` - Variable store path for the root of the BertEmbeddings model
// * `config` - `BertConfig` object defining the model architecture and vocab/hidden size
func NewBertEmbedding(p nn.Path, config BertConfig) (BertEmbeddings, error) {
	embeddingConfig := nn.DefaultEmbeddingConfig()
	embeddingConfig.PaddingIdx = 0

	//TODO: continue ...

	return BertEmbeddings{}, nil
}

/*
 *     fn new(p: &nn::Path, config: &BertConfig) -> BertEmbeddings {
 *         let embedding_config = EmbeddingConfig { padding_idx: 0, ..Default::default() };
 *
 *         let word_embeddings: nn::Embedding = embedding(p / "word_embeddings",
 *                                                        config.vocab_size,
 *                                                        config.hidden_size,
 *                                                        embedding_config);
 *
 *         let position_embeddings: nn::Embedding = embedding(p / "position_embeddings",
 *                                                            config.max_position_embeddings,
 *                                                            config.hidden_size,
 *                                                            Default::default());
 *
 *         let token_type_embeddings: nn::Embedding = embedding(p / "token_type_embeddings",
 *                                                              config.type_vocab_size,
 *                                                              config.hidden_size,
 *                                                              Default::default());
 *
 *         let layer_norm_config = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
 *         let layer_norm: nn::LayerNorm = nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);
 *         let dropout: Dropout = Dropout::new(config.hidden_dropout_prob);
 *         BertEmbeddings { word_embeddings, position_embeddings, token_type_embeddings, layer_norm, dropout }
 *     }
 *  */

// Implement BertEmbedding interface
func (be *BertEmbeddings) ForwardT(inputIds, tokenTypeIds, positionIds, inputEmbeds ts.Tensor, train bool) (ts.Tensor, string) {

	return ts.New(), ""
}
