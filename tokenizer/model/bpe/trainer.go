package bpe

// Map with no value
// Ref: https://stackoverflow.com/questions/57620170
type UintSet map[uint]struct{}

type CharSet map[rune]struct{}

type TMerge struct {
	Pair  Pair
	Count uint32
	Pos   UintSet
}

type TConfig struct {
	MinFrequency            uint32
	VocabSize               uint
	ShowProgress            bool
	SpecialTokens           []string
	LimitAlphabet           *uint
	InitialAlphabet         CharSet
	ContinuingSubwordPrefix *string
	EndOfWordSuffix         *string
}

// BpeTrainerBuilder can be used to create a `BpeTrainer`
// with a custom configuration
type BpeTrainerBuilder struct {
	Config *Config
}
