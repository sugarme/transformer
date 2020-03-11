package bpe

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"

	"github.com/emirpasic/gods/trees/binaryheap"
	// 800 stars
	// progressbar "github.com/schollz/progressbar/v2"
	// 2.2 stars
	// progressbar "github.com/cheggaaa/pb/v3"
)

// Map with no value
// Ref: https://stackoverflow.com/questions/57620170
type UintSet map[uint]struct{}

type CharSet map[string]struct{}

type TMerge struct {
	Pair  Pair
	Count uint32
	Pos   UintSet
}

// NOTE: there exists `Config`
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
	Config *TConfig
}

func NewBPETrainerBuilder() *BpeTrainerBuilder {
	config := TConfig{
		MinFrequency:            0,
		VocabSize:               30000,
		ShowProgress:            true,
		SpecialTokens:           nil,
		LimitAlphabet:           nil,
		InitialAlphabet:         nil,
		ContinuingSubwordPrefix: nil,
		EndOfWordSuffix:         nil,
	}
	return &BpeTrainerBuilder{
		Config: &config,
	}
}

// MinFequency set minimum frequency
func (btb *BpeTrainerBuilder) MinFrequency(freq uint32) {
	btb.Config.MinFrequency = freq
}

// VocabSize set the vocabulary size
func (btb *BpeTrainerBuilder) VocabSize(size uint) {
	btb.Config.VocabSize = size
}

// ShowProgress set whether to show progress
func (btb *BpeTrainerBuilder) ShowProgress(show bool) {
	btb.Config.ShowProgress = show
}

// SpecialToken set special tokens
func (btb *BpeTrainerBuilder) SpecialTokens(tokens []string) {
	btb.Config.SpecialTokens = tokens
}

//LimitAlphabet set the alphabet limit
func (btb *BpeTrainerBuilder) LimitAlphabet(limit uint) {
	btb.Config.LimitAlphabet = &limit
}

// InitialAlphabet set the initial alphabet
func (btb *BpeTrainerBuilder) InitialAlphabet(alphabet CharSet) {
	btb.Config.InitialAlphabet = alphabet
}

// ContinuingSubwordPrefix set the ContinuingSubwordPrefix
func (btb *BpeTrainerBuilder) ContinuingSubwordPrefix(prefix string) {
	btb.Config.ContinuingSubwordPrefix = &prefix
}

// EndOfWordSuffix set the EndOfWordSuffix
func (btb *BpeTrainerBuilder) EndOfWordSuffix(suffix string) {
	btb.Config.EndOfWordSuffix = &suffix
}

// Build constructs the final BpeTrainer
func (btb *BpeTrainerBuilder) Build() *BpeTrainer {
	return &BpeTrainer{
		MinFrequency:            btb.Config.MinFrequency,
		VocabSize:               btb.Config.VocabSize,
		ShowProgress:            btb.Config.ShowProgress,
		SpecialTokens:           btb.Config.SpecialTokens,
		LimitAlphabet:           btb.Config.LimitAlphabet,
		InitialAlphabet:         btb.Config.InitialAlphabet,
		ContinuingSubwordPrefix: btb.Config.ContinuingSubwordPrefix,
		EndOfWordSuffix:         btb.Config.EndOfWordSuffix,
	}
}

// BpeTrainer is in charge of training a `BPE` model from a
// mapping of words to word counts.
//
// Example:
// wordCounts := map[string]uint = {
// 	{"Hello", 1},
// 	{"World", 1},
// }
// trainer := NewBPETrainer()
// model, specialTokens := trainer.Train(wordCounts)
type BpeTrainer struct {
	// The minimum frequency a pair must have to produce a merge operation
	MinFrequency uint32
	// The target vocabulary size
	VocabSize uint
	// Whether to show progress while training
	ShowProgress bool
	// A list of special tokens that the model should know of
	SpecialTokens []string
	// Whether to limit the number of initial tokens that can be kept before
	// computing merges
	LimitAlphabet *uint
	// The initial alphabet we want absolutely to include. This allows to cover
	// some characters that are not necessarily in the training set
	InitialAlphabet CharSet
	// An optional prefix to use on any subword that exist only behind another one
	ContinuingSubwordPrefix *string
	// An optional suffix to characterize and end-of-word subword
	EndOfWordSuffix *string
}

func NewBpeTrainer(minFreq uint32, vocabSize uint) *BpeTrainer {
	btb := NewBPETrainerBuilder()
	bpeTrainer := btb.Build()

	bpeTrainer.MinFrequency = minFreq
	bpeTrainer.VocabSize = vocabSize

	return bpeTrainer

}

func (bt *BpeTrainer) setupProgress() interface{} {
	if bt.ShowProgress {
		// TODO: setup progress bar
	}
	return nil
}

// set the progress bar in the finish state
func (bt *BpeTrainer) finalizeProgress(pb interface{}, finalLen uint) interface{} {
	if pb != nil {
		// TODO:
		// set length
		// finish up
	}

	return nil
}

// updateProgress update the progress bar with the new provided length and msg
func (bt *BpeTrainer) updateProgress(p interface{}, len uint, msg string) {
	// TODO: update progress bar
}

// addSpecialTokens adds the provided special tokens to the initial vocabulary
func (bt *BpeTrainer) addSpecialTokens(w2id map[string]uint32, id2w []string) {
	for _, tok := range bt.SpecialTokens {
		if _, ok := w2id[tok]; !ok {
			id2w = append(id2w, tok)
			w2id[tok] = uint32(len(id2w) - 1)
		}
	}
}

// computeAlphabet computes the initial alphabet and limit it if relevant
func (bt *BpeTrainer) computeAlphabet(wc, w2id map[string]uint32, id2w []string) {
	// compute the alphabet from seen words
	var alphabet map[string]uint = make(map[string]uint)

	for word, count := range wc {
		chars := strings.Split(word, "")
		for _, char := range chars {
			var newCount uint = 0
			// if char not existing, newCount will be zero
			if newCount, ok := alphabet[char]; ok {
				newCount += uint(count)
			}
			alphabet[char] = newCount
		}
	}

	// Also, include anything from the provided intial alphabet
	// NOTE: InitialAlphabet is CharSet which is map[string]struct{}
	for initChar, _ := range bt.InitialAlphabet {
		// asign a uint max as frequency
		alphabet[initChar] = math.MaxUint32
	}

	type keptItem struct {
		Char string
		Freq uint
	}
	var kept []keptItem
	for char, freq := range alphabet {
		kept = append(kept, keptItem{char, freq})
	}

	// compute the number of chars to remove from the alphabet
	// if `limitAlphabet` < `len(initialAlphabet)` some of these
	// initial characters will be removed.
	var toRemove int = 0
	var limit int
	if bt.LimitAlphabet != nil {
		limit = int(*bt.LimitAlphabet)
		if len(alphabet) > int(*bt.LimitAlphabet) {
			toRemove = len(alphabet) - limit
		}
	}

	// remove the unwanted `chars`
	if toRemove > 0 {
		// 1. Sort kept by char alphabetically?
		// TODO: double-check this (sort by char or freq? asc or desc)
		sort.Slice(kept, func(i, j int) bool {
			return kept[i].Char < kept[j].Char
		})
		// 2. Remove the unwanted chars
		kept = kept[:toRemove]
	}

	// Keep the initial alphabet (sorted by determinism)
	sort.Slice(kept, func(i, j int) bool {
		// sort by freq
		return kept[i].Freq < kept[j].Freq
	})

	for _, k := range kept {
		if _, ok := w2id[k.Char]; !ok {
			id2w = append(id2w, k.Char)
			w2id[k.Char] = uint32(len(id2w) - 1)
		}
	}
}

// tokenizerWord tokenizes words and adds subwords to the vocabulary when relevant
func (bt *BpeTrainer) tokenizeWords(wc map[string]uint32, w2id map[string]uint32, id2w []string, pb interface{}) ([]Word, []uint32) {
	// NOTE: bp is progress bar.
	// TODO: update bp to specific progress bar type

	// words := make([]Word, len(wc))
	// counts := make([]uint32, len(wc))
	var words []Word
	var counts []uint32

	for word, count := range wc {
		var currentWord Word
		counts = append(counts, count)

		chars := strings.Split(word, "")

		for i, c := range chars {
			var s string
			if _, ok := w2id[c]; ok {
				// Found the initial char in the authorized alphabet
				// Add the `continuingSubwordPrefix` if relevant
				if i > 0 && i < len(chars)-1 {
					if prefix := bt.ContinuingSubwordPrefix; prefix != nil {
						s = fmt.Sprintf("%v%v", &prefix, c)
					} else {
						s = c
					}
				}
				// Add the `endOfWordSuffix` if relevant
				if i == len(chars)-1 { // last `char`
					if suffix := bt.EndOfWordSuffix; suffix != nil {
						s = fmt.Sprintf("%v%v", &suffix, c)
					} else {
						s = c
					}
				}

				// Insert the new formed string if neccessary
				if _, ok := w2id[s]; !ok {
					id2w = append(id2w, s)
					w2id[s] = uint32(len(id2w) - 1)
				}

				fmt.Println(w2id[s])

				currentWord.Add(w2id[s])

			}
		} // end loop of `chars`

		words = append(words, currentWord)

		// TODO: update progress bar to 1

	} // end loop of `wc`

	return words, counts

}

// countPairs counts ...
func (bt *BpeTrainer) countPairs(words []Word, counts []uint32, progress interface{}) (map[Pair]uint32, map[Pair]UintSet) {

	var pairCounts map[Pair]uint32 = make(map[Pair]uint32, bt.VocabSize*2)
	var whereToUpdate map[Pair]UintSet = make(map[Pair]UintSet, bt.VocabSize*2)

	// Divide w into work units that take ~100μs-1ms to compute.
	n := len(words)
	size := int(1000000 / n)
	if size < 1 {
		size = 1
	}

	var wg sync.WaitGroup
	wg.Add(n)
	for i, j := 0, size; i < n; i, j = j, j+size {
		if j > n {
			j = n
		}

		// wg.Add(1)

		go func(i, j int) {
			for k := i; k < j; k++ {
				// Do individual task here with index k
				word := words[k]

				var window = 2
				for x := 0; i < len(word.Symbols)-1; x += window - 1 {
					y := x + window
					if y >= len(word.Symbols) {
						y = len(word.Symbols)
					}

					w := word.Symbols[x:y]
					pair := Pair{
						C1: w[0].C,
						C2: w[1].C,
					}

					// Initialize pairCounts and whereToUpdate for this pair
					// if we just seen it
					if _, ok := pairCounts[pair]; !ok {
						pairCounts[pair] = 0
					}

					// Then update counts
					count := counts[k]
					// hashset map[uint]struct{}
					var hs UintSet
					if h, ok := whereToUpdate[pair]; ok {
						h[uint(k)] = struct{}{} // found. Modify it
					} else {
						// create a new
						hs[uint(k)] = struct{}{}
						whereToUpdate[pair] = hs
					}

					pairCounts[pair] += count
				}

				// TODO: update progress bar

				wg.Done()
			}

		}(i, j)
	}

	wg.Wait()

	// Aggregate results

	// TODO: test whether having a data race??? as goroutines update pairCounts and whereToUpdate

	return pairCounts, whereToUpdate

}

func (bt *BpeTrainer) Train(wordCounts map[string]uint32) (BPE, []string) {
	var (
		wordToId map[string]uint32 = make(map[string]uint32)
		idToWord []string
	)

	var progress = bt.setupProgress()

	// 1. Add all special tokens to the vocabular
	bt.addSpecialTokens(wordToId, idToWord)

	// 2. Compute the initial alphabet
	bt.computeAlphabet(wordCounts, wordToId, idToWord)

	// 3. Tokenize words
	bt.updateProgress(progress, uint(len(wordCounts)), "Tokenize word")

	// fmt.Println(wordCounts)
	// fmt.Println(wordToId)
	// fmt.Println(idToWord)

	words, counts := bt.tokenizeWords(wordCounts, wordToId, idToWord, progress)

	bt.finalizeProgress(progress, uint(len(words)))
	fmt.Println(words)
	// fmt.Println(counts)

	// 4. Count pairs in words
	bt.updateProgress(progress, uint(len(words)), "Count pairs")

	var (
		pairCounts    map[Pair]uint32  = make(map[Pair]uint32)
		whereToUpdate map[Pair]UintSet = make(map[Pair]UintSet)
	)

	pairCounts, whereToUpdate = bt.countPairs(words, counts, progress)

	// insert them to the queue
	var queue = binaryheap.NewWithIntComparator()

	for pair, pos := range whereToUpdate {
		if count, ok := pairCounts[pair]; ok {
			queue.Push(TMerge{
				Pair:  pair,
				Count: count,
				Pos:   pos,
			})
		}
	}
	bt.finalizeProgress(progress, uint(len(words)))

	// 5. Do merges
	bt.updateProgress(progress, bt.VocabSize, "Compute merges")

	type TMerges struct {
		Pair    Pair
		PairVal uint32
	}

	var merges []TMerges

	for {
		// Stop as soon as we have a big enough vocabulary
		if uint(len(wordToId)) >= bt.VocabSize {
			break
		}

		if queue.Empty() {
			break
		}

		t, _ := queue.Pop()
		var top TMerge = t.(TMerge)
		if top.Count != pairCounts[top.Pair] {
			top.Count = pairCounts[top.Pair]
			queue.Push(top)

			continue
		}

		if top.Count < 1 || bt.MinFrequency > top.Count {
			break
		}

		partA := idToWord[*top.Pair.C1]
		partB := idToWord[*top.Pair.C2]

		// Build new token
		if prefix := bt.ContinuingSubwordPrefix; prefix != nil {
			if strings.HasPrefix(partB, *prefix) {
				// strip prefix
				partB = strings.TrimPrefix(partB, *prefix)
			}
		}

		newToken := fmt.Sprintf("%v%v", partA, partB)

		// Insert new token
		newTokenId := uint32(len(idToWord))

		idToWord = append(idToWord, newToken)
		wordToId[newToken] = newTokenId
		merges = append(merges, TMerges{top.Pair, newTokenId})

		// Merge the new pair in every words
		type TChange struct {
			WChange WChange
			WIndex  int
		}
		var changes []TChange
		for i, _ := range top.Pos {
			// NOTE: words []Word
			// TODO: merge each of these words concurrently
			w := words[i]
			wChanges, err := w.Merge(*top.Pair.C1, *top.Pair.C2, newTokenId)
			if err != nil {
				fmt.Println(err)
			}
			for _, wc := range wChanges {
				changes = append(changes, TChange{wc, int(i)})
			}

		}

		// Introduce new formed pairs
		for _, tc := range changes {
			count := tc.WChange.Change
			pair := Pair{tc.WChange.C1, tc.WChange.C2}

			c, _ := pairCounts[pair]
			c += uint32(count)
			pairCounts[pair] = c

			if tc.WChange.Change > 0 {
				var hs UintSet
				if h, ok := whereToUpdate[pair]; ok {
					h[uint(tc.WIndex)] = struct{}{}
					hs = h
				} else {
					// if not existing, we create new one anyway
					h[uint(tc.WIndex)] = struct{}{}
					hs = h
				}

				whereToUpdate[pair] = hs
			}
		}

		for pair, pos := range whereToUpdate {
			count := pairCounts[pair]

			if count > 0 {
				queue.Push(TMerge{
					pair, count, pos,
				})
			}
		}

		// TODO: update progress bar by 1

	} // end of `for` loop

	bt.finalizeProgress(progress, uint(len(merges)))

	var builder *BpeBuilder
	builder = NewBpeBuilder()

	var newMerges Merges
	for i, m := range merges {
		pairVal := PairVal{
			uint32(i),
			m.PairVal,
		}
		newMerges[m.Pair] = pairVal
	}

	builder.VocabAndMerges(wordToId, newMerges)

	if prefix := bt.ContinuingSubwordPrefix; prefix != nil {
		builder.ContinuingSubwordPrefix(*prefix)
	}

	if suffix := bt.EndOfWordSuffix; suffix != nil {
		builder.EndOfWordSuffix(*suffix)
	}

	bpe, err := builder.Build()
	if err != nil {
		fmt.Println(err)
	}

	return *bpe, bt.SpecialTokens

}

// implement Trainer interface for BpeTrainer

// Train a BPE model
func (bt *BpeTrainer) train(wordCounts map[string]uint32) (BPE, []string) {
	return bt.Train(wordCounts)
}

// Process a bunch of toke, counting them
func (bt *BpeTrainer) processTokens(words map[string]uint32, tokens []string) {
	for _, token := range tokens {
		c, _ := words[token]
		c += 1
		words[token] = c
	}
}

// Whether we should show progress
func (bt *BpeTrainer) shouldShowProgress() bool {
	return bt.ShowProgress
}
