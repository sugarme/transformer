package bpe

type Vocab map[string]uint32
type VocabR map[uint32]string
type Merges map[Pair]PairVal

type ConfigFiles struct {
	Vocab  string
	Merges string
}

type Config struct {
	Files                   *ConfigFiles
	Vocab                   Vocab
	Merges                  Merges
	CacheCapacity           uint
	Dropout                 *float32
	UnkToken                *string
	ContinuingSubwordPrefix *string
	EndOfWordSuffix         *string
}

// BpeBuilder can be used to create a `BPE` model with
// a custom configuration
type BpeBuilder struct {
	Config Config
}

func NewBpeBuilder() *BpeBuilder {
	return &BpeBuilder{
		Config: Config{
			Files:                   nil,
			Vocab:                   *new(Vocab),
			Merges:                  *new(Merges),
			CacheCapacity:           DefaultCacheCapacity,
			Dropout:                 nil,
			UnkToken:                nil,
			ContinuingSubwordPrefix: nil,
			EndOfWordSuffix:         nil,
		},
	}
}

// Files sets input files for the model
func (bb *BpeBuilder) Files(vocab string, merges string) {
	bb.Config.Files = &ConfigFiles{vocab, merges}
}

// VocabAndMerges sets vocab and merges
func (bb *BpeBuilder) VocabAndMerges(vocab Vocab, merges Merges) {
	bb.Config.Vocab = vocab
	bb.Config.Merges = merges
}

// CacheCapacity sets the cache capacity. Disable cache by setting it to 0
func (bb *BpeBuilder) CacheCapacity(capacity uint) {
	bb.Config.CacheCapacity = capacity
}

// Dropout set dropout for model
// Ref. https://arxiv.org/abs/1910.13267
func (bb *BpeBuilder) Dropout(dropout float32) {
	bb.Config.Dropout = &dropout
}
