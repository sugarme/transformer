package bpe

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"strconv"
	"strings"
)

type Vocab map[string]uint32
type VocabR map[uint32]string
type Merges map[Pair]PairVal

type ConfigFiles struct {
	Vocab  string
	Merges string
}

type Config struct {
	Files                   *ConfigFiles
	Vocab                   *Vocab
	Merges                  *Merges
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
	bb.Config.Vocab = &vocab
	bb.Config.Merges = &merges
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

// UnkToken set the `UNK` token for the vocab
func (bb *BpeBuilder) UnkToken(unkTok string) {
	bb.Config.UnkToken = &unkTok
}

// ContinuingSubword set the `continuingSubwordPrefix` option.
func (bb *BpeBuilder) ContinuingSubwordPrefix(continuingSubwordPrefix string) {
	bb.Config.ContinuingSubwordPrefix = &continuingSubwordPrefix
}

// EndOfWordSuffix set the `endOfWordSuffix` option.
func (bb *BpeBuilder) EndOfWordSuffix(endOfWordSuffix string) {
	bb.Config.EndOfWordSuffix = &endOfWordSuffix
}

// Build returns a `BPE` model that uses the BpeBuilder configuration
func (bb *BpeBuilder) Build() (*BPE, error) {
	var (
		err    error
		vocab  Vocab
		merges Merges
		vocabR VocabR
		cache  *Cache
		bpe    BPE
	)
	// validate dropout
	if p := *bb.Config.Dropout; &p != nil {
		if p <= 0.0 || p > 1.0 {
			err = errors.New("Error: Invalid dropout.")
			return nil, err
		}
	}

	// Read files if provided
	if bb.Config.Files != nil {
		vocab, merges = BPE.ReadFiles(&bb.Config.Files.Vocab, bb.Config.Files.Merges)
		bb.Config.Vocab = &vocab
		bb.Config.Merges = &merges

		for k, v := range vocab {
			vocabR[v] = k
		}
	}
	if bb.Config.CacheCapacity != 0 {
		cache = NewCache(bb.Config.CacheCapacity)
	} else {
		cache = nil
	}

	bpe = BPE{
		Vocab:                   &vocab,
		VocabR:                  &vocabR,
		Merges:                  &merges,
		Cache:                   cache,
		Dropout:                 bb.Config.Dropout,
		UnkToken:                bb.Config.UnkToken,
		ContinuingSubwordPrefix: bb.Config.ContinuingSubwordPrefix,
		EndOfWordSuffix:         bb.Config.EndOfWordSuffix,
	}

	return &bpe, nil

}

// BPE is a struct for byte pair encoding model
// Ref. https://www.aclweb.org/anthology/P16-1162/
type BPE struct {
	// Vocab is the vocabulary assigns a number to each token.
	Vocab *Vocab

	// VocabR is Reversed vocabulary, to rebuild sentences.
	VocabR *VocabR

	// Merges contains the mapping between Pairs and their (rank, newId).
	Merges *Merges

	// Cache contains the cache for optimizing the encoding step.
	// It is a `map[string]Word`
	Cache *Cache

	// Dropout probability for merges.
	// 0 = no dropout is the default.
	// At 1.0, tokenization will perform no merges, so the result will just be characters.
	Dropout *float32

	// UnkToken is the unknown token to be used when we encounter an unknown char
	UnkToken *string

	// ContinuingSubwordPrefix is an optional prefix
	// to use on any subword that exist only behind another one
	ContinuingSubwordPrefix *string

	// EndOfWordSuffix is an optional suffix
	// to caracterize and end-of-word subword
	EndOfWordSuffix *string
}

func (b *BPE) builder() *BpeBuilder {
	return NewBpeBuilder()
}

// new create a BPE with default values
func (b *BPE) new() {
	// TODO: handling error
	b, _ = b.builder().Build()
}

// `Clone` can't be derive because it's not implemented for `Cache`.
// To keep things simple when we clone, the new BPE will start with a fresh cache.
func (b *BPE) clone() {
	newBpe := b
	newBpe.Cache.Fresh()
	b = newBpe
}

// New creates new BPE model with given vocab and merges
func (b *BPE) New(vocab Vocab, merges Merges) {
	b.new()
	b.Vocab = &vocab
	b.Merges = &merges
}

// FromFile creates `BpeBuilder` from vocab and merges files.
func (b *BPE) FromFile(vocab string, merges string) *BpeBuilder {
	builder := b.builder()
	builder.Files(vocab, merges)
	return builder
}

// ReadFiles read the given files to extract vocab and merges
func (b *BPE) ReadFiles(vocabF string, mergesF string) (*Vocab, *Merges, error) {
	var err error
	// read json file
	vocabBytes, err := ioutil.ReadFile(vocabF)
	if err != nil {
		return nil, nil, err
	}

	var (
		vocab  Vocab
		merges Merges
	)

	err = json.Unmarshal(vocabBytes, &vocab)
	if err != nil {
		return nil, nil, err
	}

	// Read merges file. Each line contains a Merges object(rank, )
	// Recall: Merges is map[Pair]PairVal (rank uint32, newId uint32)
	mFile, err := os.Open(mergesF)
	if err != nil {
		return nil, nil, err
	}
	defer mFile.Close()

	s := bufio.NewScanner(mFile)

	// `s.Scan()` advance scaning and return `false` if
	// end of file or hit any error. The error will be
	// access by s.Err. If error caused by EOF it's value is nil.
	var lineNum = 0
	for s.Scan() {
		line := s.Text()

		// Skip line with `#version`
		re := regexp.MustCompile(`#version`)
		if re.MatchString(line) {
			continue
		}

		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			err = fmt.Errorf("Merges file error: invalid data at line %d\n", lineNum)
			return nil, nil, err
		}

		a, ok := vocab[parts[0]]
		if !ok {
			err = fmt.Errorf("Error: value for %s key not found.", parts[0])
			return nil, nil, err
		}

		b, ok := vocab[parts[1]]
		if !ok {
			err = fmt.Errorf("Error: value for %s key not found.", parts[0])
			return nil, nil, err
		}

		pair := Pair{&a, &b}
		newToken := fmt.Sprint("%v%v", parts[0], parts[1])
		newId, ok := vocab[newToken]
		if !ok {
			err = fmt.Errorf("Error: value for %s key not found.", parts[0])
			return nil, nil, err
		}

		newTokenInt, err := strconv.ParseInt(newToken, 10, 64)
		if err != nil {
			return nil, nil, err
		}

		pairVal := PairVal{uint32(newTokenInt), newId}

		merges[pair] = pairVal

		lineNum += 1

	}

	if s.Err != nil {
		return nil, nil, s.Err()
	}

	return &vocab, &merges, nil

}
