package tokenizer

// tokenizer represents a tokenization pipeline
// TODO: full description

import (
	"fmt"
	"log"
	// "path/filepath"
	"bufio"
	"context"
	"os"
	"regexp"
	"strings"
	"sync"

	progressbar "github.com/schollz/progressbar/v2"
	"golang.org/x/sync/errgroup"

	"github.com/sugarme/sermo/normalizer"
	"github.com/sugarme/sermo/util"
)

type Offsets struct {
	Start uint
	End   uint
}

type PreToken struct {
	Value   string
	Offsets Offsets
}

type Token struct {
	Id      uint32
	Value   string
	Offsets Offsets
}

// PreTokenizer processes strings before going to the model
type PreTokenizer interface {
	PreTokenize(s string) []PreToken
}

// Model represents a model used during tokenization (i.e., BPE, Word, or Unigram)
type Model interface {
	Tokenize(tokens []PreToken) ([]Token, error)
	TokenToId(token string) uint32
	IdToToken(id uint32) string
	GetVocabSize() uint
	Save(path string, name string) error
}

type PostProcessor interface {
	// Returns the number of tokens that will be added during the processing step
	AddedTokens(isPair bool) uint
	// Process processes both encodings and returns a new merged one
	// NOTE: pairEncoding is optional
	Process(...Encoding) Encoding
}

// Decoder takes care of (merges) the given slice of tokens to string
type Decoder interface {
	Decode(tokens []string) string
}

// Trainer is responsible for training a model. It takes lines/sentences
// and returns a tokenizer `Model` when done.
type Trainer interface {
	// Whether showing progress bar or not
	WithProgressBar() bool
	// Actual training method. It will return a trained model and
	// a list of `special tokens` to be added directly to the tokenizer
	// along with the model
	Train(words map[string]uint32) (Model, []string)
	// ProcessTokens processes a bunch of tokens and counts them as relevant
	ProcessTokens(words map[string]uint32, tokens []string)
}

// Implement methods for `Token`
// NewToken generate new token from input data
func NewToken(id uint32, value string, offsets Offsets) Token {
	return Token{
		Id:      id,
		Value:   value,
		Offsets: offsets,
	}
}

type Single struct {
	Sentence string
}
type Dual struct {
	Sentence string
	Pair     string
}

type EncodeInput interface{}

// var EncodeInput map[interface{}]string = map[interface{}]string{
// "Single": "Single",
// Dual{
// Value: "Value",
// Pair:  "Pair",
// }: "Dual",
// }

type AddedToken struct {
	Content      string // content of the added token
	IsSingleWord bool   // whether this token is single word or break words
}

// AddedTokenFrom creates AddedToken from input content
func AddedTokenFrom(content string, opt ...bool) AddedToken {
	var isSingleWord bool = false
	if len(opt) > 0 {
		isSingleWord = opt[0]
	}
	return AddedToken{
		Content:      content,
		IsSingleWord: isSingleWord,
	}
}

// Default initiates an addedtoken with default value
func (at *AddedToken) Default() {
	at = &AddedToken{
		Content:      "",
		IsSingleWord: false,
	}
}

// Hash hashes on the content of addedtoken
// Should we use SipHash or use map?
func (at *AddedToken) Hash() {

}

// Eq checks whether current addedtoken content is equal to other
func (at *AddedToken) Eq(other AddedToken) bool {
	return at.Content == other.Content
}

// Tokenizer represents a tokenization pipeline.
// It can implement any encoding or decoding of any text.
type Tokenizer struct {
	// Parts
	Normalizer    *normalizer.Normalizer
	PreTokenizer  *PreTokenizer
	Model         *Model
	PostProcessor *PostProcessor
	Decoder       *Decoder

	// Vocab
	AddedTokens   map[AddedToken]uint32
	AddedTokensR  map[uint32]AddedToken
	SplitRe       *regexp.Regexp
	SpecialTokens map[string]uint32

	// General processing parameters
	Trunc   *TruncationParams
	Padding *PaddingParams
}

// Implementing methods for Tokenizer
func NewTokenizer(model Model) Tokenizer {
	return Tokenizer{
		Normalizer:    nil,
		PreTokenizer:  nil,
		Model:         &model,
		PostProcessor: nil,
		Decoder:       nil,

		AddedTokens:   make(map[AddedToken]uint32),
		AddedTokensR:  make(map[uint32]AddedToken),
		SplitRe:       &regexp.Regexp{},
		SpecialTokens: make(map[string]uint32),

		Trunc:   nil,
		Padding: nil,
	}
}

func (t *Tokenizer) WithNormalizer(n normalizer.Normalizer) {
	t.Normalizer = &n
}

func (t *Tokenizer) GetNormalizer() normalizer.Normalizer {
	return *t.Normalizer
}

func (t *Tokenizer) WithPreTokenizer(preTokenizer PreTokenizer) {
	t.PreTokenizer = &preTokenizer
}

func (t *Tokenizer) GetPreTokenizer() PreTokenizer {
	return *t.PreTokenizer
}

func (t *Tokenizer) WithPostProcessor(postProcessor PostProcessor) {
	t.PostProcessor = &postProcessor
}

func (t *Tokenizer) GetPostProcessor() PostProcessor {
	return *t.PostProcessor
}

func (t *Tokenizer) WithDecoder(decoder Decoder) {
	t.Decoder = &decoder
}

func (t *Tokenizer) GetDecoder() Decoder {
	return *t.Decoder
}

func (t *Tokenizer) WithModel(model Model) {
	t.Model = &model
}

func (t *Tokenizer) GetModel() Model {
	return *t.Model
}

func (t *Tokenizer) WithTruncation(trunc TruncationParams) {
	t.Trunc = &trunc
}

func (t *Tokenizer) GetTruncation() TruncationParams {
	return *t.Trunc
}

func (t *Tokenizer) WithPadding(padding PaddingParams) {
	t.Padding = &padding
}

func (t *Tokenizer) GetVocabSize(withAddedToken bool) uint {
	if withAddedToken {
		return (*t.Model).GetVocabSize() + uint(len(t.AddedTokens))
	}

	return (*t.Model).GetVocabSize()
}

func (t *Tokenizer) TokenToId(token string) uint32 {

	addedToken := AddedTokenFrom(token)

	id, ok := t.AddedTokens[addedToken]
	if ok {
		return id
	}

	return (*t.Model).TokenToId(token)

}

func (t *Tokenizer) IdToToken(id uint32) string {
	tok, ok := t.AddedTokensR[id]
	if ok {
		return tok.Content
	}
	return (*t.Model).IdToToken(id)
}

func (t *Tokenizer) NumAddedTokens(isPair bool) uint {
	return (*t.PostProcessor).AddedTokens(isPair)
}

type splitRes struct {
	Content string
	Id      uint32
	Found   bool // whether split was found in AddedTokens/SpecialTokens
}

// Encode encodes the given sentence
func (t *Tokenizer) Encode(input EncodeInput) Encoding {
	generateOutput := func(sentence string, typeId uint32) Encoding {
		// Split into as many sequences as needed to avoid splitting
		// on our added tokens
		var splits []splitRes
		var encodings []Encoding

		splits = t.splitOnAddedTokens(sentence)
		for _, s := range splits {
			// If this is one of our added tokens, return an encoding directly
			if s.Found {
				e := NewEncoding(*normalizer.NewNormalizedFrom(s.Content), []uint32{s.Id}, []uint32{typeId}, []string{s.Content}, []Offsets{{0, uint(len(s.Content))}}, []uint32{0}, []uint32{1}, []Encoding{})

				encodings = append(encodings, e)
			}

			// 1. Normalization
			var normalized *normalizer.Normalized
			if t.Normalizer != nil {
				normalized = normalizer.NewNormalizedFrom(s.Content)
			}

			// 2. Pre-tokenization
			var preTokenized []PreToken

			if t.PreTokenizer != nil {
				preTokenized = (*t.PreTokenizer).PreTokenize(normalized.Get().Normalized)
			}

			// 3. Model
			output, _ := (*t.Model).Tokenize(preTokenized)

			var en Encoding

			for _, t := range output {
				en.Ids = append(en.Ids, t.Id)
				en.Tokens = append(en.Tokens, t.Value)
				en.Offsets = append(en.Offsets, t.Offsets)
			}

			for i := range output {
				en.TypeIds[i] = typeId
				en.SpecialTokenMask[i] = 0
				en.AttentionMask[i] = 1
			}

			en.Overflowing = []Encoding{}

			encodings = append(encodings, en)
		} // end loop over splits

		if len(encodings) == 0 {
			return DefaultEncoding()
		}

		// split off at position 1
		first := encodings[0]
		others := encodings[1:]

		// Put others to overflowing of first
		for _, e := range others {
			first.MergeWith(e)
		}

		return first
	} // end of anonymous function `generateOutput`

	var (
		sentence, pair         string
		encoding, pairEncoding Encoding
	)
	switch input.(type) {
	case Single:
		sentence = input.(Single).Sentence
	case Dual:
		sentence = input.(Dual).Sentence
		pair = input.(Dual).Pair
	}

	encoding = generateOutput(sentence, 0)

	if len(pair) > 0 {
		pairEncoding = generateOutput(pair, 1)
	}

	// 4. Post processing
	if t.PostProcessor != nil {
		return (*t.PostProcessor).Process(encoding, pairEncoding)
	}

	encoding = t.postProcess(encoding, pairEncoding)

	// NOTE.Should we return pairEncoding as well?
	return encoding
}

// EncodeBatch encodes all sentences in concurrency
func (t *Tokenizer) EncodeBatch(inputs []EncodeInput) []Encoding {
	var encodings []Encoding
	var wg sync.WaitGroup

	wg.Add(len(inputs))

	// Encoding concurrently
	for i := 0; i < len(inputs); i++ {
		go func(i int) {
			defer wg.Done()

			e := t.Encode(inputs[i])
			encodings = append(encodings, e)

		}(i)
	}

	wg.Wait()

	// Do padding if included
	if t.Padding != nil {
		PadEncodings(encodings, *t.Padding)
	}

	return encodings
}

// Decode returns a corresponding string from an input id
func (t *Tokenizer) Decode(ids []uint32, skipSpecialTokens bool) string {
	var tokens []string

	for _, id := range ids {
		// Look up at added tokens
		var token string
		tok, ok := t.AddedTokensR[id]
		if !ok {
			// Look up at model
			token = t.IdToToken(id)
		}

		token = tok.Content

		_, ok = t.SpecialTokens[token]

		if !skipSpecialTokens || !ok {
			tokens = append(tokens, token)
		}
	}

	if t.Decoder != nil {
		return (*t.Decoder).Decode(tokens)
	}

	return strings.Join(tokens, " ")
}

// DecodeBatch decodes all sentences in concurrency
func (t *Tokenizer) DecodeBatch(sentences [][]uint32, skipSpecialTokens bool) []string {
	var decodings []string
	var wg sync.WaitGroup

	wg.Add(len(sentences))

	// Decoding concurrently
	for i := 0; i < len(sentences); i++ {
		go func(i int) {
			defer wg.Done()

			s := t.Decode(sentences[i], skipSpecialTokens)
			decodings = append(decodings, s)

		}(i)
	}

	wg.Wait()

	return decodings
}

func (t *Tokenizer) splitOnAddedTokens(sentence string) []splitRes {

	var splits []splitRes
	rs := []rune(sentence)
	var allSplits [][]int

	// if there's no splitRe (regular epxression to split), do nothing
	if t.SplitRe == nil {
		splits = append(splits, splitRes{sentence, 0, false})
		return splits
	}

	// matches contains slice of 2-element items (start and end byte position)
	// of the matched strings
	matches := t.SplitRe.FindAllStringIndex(sentence, -1)

	for _, m := range matches {
		splits = append(splits, splitRes{
			Content: string(rs[m[0]:m[1]]),
			Id:      0,
		})
	}

	// Collect also the splits in-between added tokens
	startOffset := 0
	for _, m := range matches {
		if startOffset < m[0] {
			allSplits = append(allSplits, []int{startOffset, m[0]})
		}

		allSplits = append(allSplits, []int{m[0], m[1]})
		startOffset = m[1]
	}

	// Check for the last piece
	last := allSplits[len(allSplits)]
	if last[1] < len(sentence) {
		allSplits = append(allSplits, []int{last[1], len(sentence)})
	}

	if len(allSplits) == 0 {
		splits = append(splits, splitRes{sentence, 0, false})
		return splits
	}

	for _, m := range allSplits {
		s := string(rs[m[0]:m[1]])
		// Look up at special tokens
		id, ok := t.SpecialTokens[s]
		// not found. Look up at added tokens
		if !ok {
			// If not found, id will be 0 and ok = false
			id, ok = t.AddedTokens[AddedToken{
				Content:      s,
				IsSingleWord: false,
			}]
			if !ok {
				splits = append(splits, splitRes{
					Content: s,
					Id:      0,
					Found:   false,
				})
			}
		}
		splits = append(splits, splitRes{
			Content: s,
			Id:      id,
			Found:   true,
		})
	}

	return splits

}

// Train trains a model and replaces the current model using a given trainer
func (t *Tokenizer) Train(trainer Trainer, files []string) error {
	type Job struct {
		File     string
		Progress *progressbar.ProgressBar
	}

	var jobs []Job

	for _, f := range files {
		fsize, err := util.FileSize(f)
		if err != nil {
			log.Fatal(err)
		}
		bar := progressbar.New(int(fsize))

		jobs = append(jobs, Job{f, bar})
	}

	// Doing jobs concurrently

	g, ctx := errgroup.WithContext(context.Background())
	lnChan := make(chan map[string]uint32)

	for i := 0; i < len(jobs); i++ {
		current := i
		g.Go(func() error {
			// Now, do the job
			file, err := os.Open(jobs[current].File)
			if err != nil {
				return err
			}
			defer file.Close()

			var line string
			words := make(map[string]uint32)

			scanner := bufio.NewScanner(file)
			for scanner.Scan() {
				line = scanner.Text()
				// io.scanner returns line w/o `\n`. We add it back manually.
				line = fmt.Sprintf("%v\n", line)

				normalized := t.normalize(line)
				preTokenized := t.preTokenize(normalized.Normalized)
				var tokens []string
				for _, tok := range preTokenized {
					tokens = append(tokens, tok.Value)
				}
				trainer.ProcessTokens(words, tokens)

				// Pass processed data to channel
				lnChan <- words

				select {
				case lnChan <- words:
					// Keep going
				case <-ctx.Done():
					return ctx.Err()
				}
			}

			if err := scanner.Err(); err != nil {
				return err
			}

			return nil

		})
	}

	// Close out the channel when the first error occurs or
	// when processing is successful.
	go func() {
		g.Wait()
		close(lnChan)
	}()

	// Handle result coming from channel
	// words is a dictionary of words and their frequency
	words := make(map[string]uint32)

	// calculate frequency and create a final map
	for result := range lnChan {
		for w, c := range result {
			count, ok := words[w]
			// word exists, sum up frequency
			if ok {
				words[w] = count + c
			}
			// word not exist, let add it
			words[w] = c
		}
	}

	// Training model
	model, specialTokens := trainer.Train(words)

	// Replace with trained model
	t.Model = &model
	t.AddSpecialTokens(specialTokens)

	// as long as an error occurs, return it.
	return g.Wait()
}

// PreTokenize processes logic, handling the case where there is no PreTokenizer set
func (t *Tokenizer) preTokenize(sentence string) []PreToken {
	if t.PreTokenizer == nil {
		return []PreToken{
			{
				Value:   sentence,
				Offsets: Offsets{0, uint(len(sentence))},
			},
		}
	}
	return (*t.PreTokenizer).PreTokenize(sentence)
}

// normalize normalizes using given normalizer
func (t *Tokenizer) normalize(sequence string) normalizer.NormalizedString {
	normalized := normalizer.NewNormalizedFrom(sequence)
	return normalized.Get()
}

// AddSpecialTokens registers give tokens as special tokens. This is especially useful
// for removing them while decoding.
func (t *Tokenizer) AddSpecialTokens(tokens []string) uint {
	var addedTokens []AddedToken
	for _, tok := range tokens {
		addedTok := AddedTokenFrom(tok)
		addedTokens = append(addedTokens, addedTok)

		// add to special tokens
		id := t.TokenToId(tok)
		if id > 0 {
			t.SpecialTokens[tok] = id
		}
	}

	added := t.AddTokens(addedTokens)

	t.refreshAddedTokens()

	return added

}

// AddTokens adds given tokens to added vocabulary
func (t *Tokenizer) AddTokens(tokens []AddedToken) uint {
	var ignored = 0
	for _, tok := range tokens {
		ok := t.TokenToId(tok.Content)
		if len(tok.Content) == 0 || ok > 0 {
			ignored += 1
			continue
		}

		newId := uint32((*t.Model).GetVocabSize()) + uint32(len(t.AddedTokens))
		id := t.AddedTokens[tok]
		// found
		if id > 0 {
			ignored += 1
		}
		// not found. Add it
		t.AddedTokens[tok] = newId
		// update the current revert map
		t.AddedTokensR[newId] = tok

	}

	t.refreshAddedTokens()

	// Return the number of added tokens
	return uint(len(tokens) - ignored)
}

// PostProcess processes the case where there is no PostProcessor set
func (t *Tokenizer) postProcess(encoding Encoding, pairEncodings ...Encoding) Encoding {

	var (
		isPaired     bool = false
		pairEncoding Encoding
	)
	// 1. Truncate if needed
	if t.Trunc != nil {
		var nAddedTokens uint
		if t.PostProcessor == nil {
			nAddedTokens = 0
		}

		if pairEncodings != nil {
			isPaired = true
			pairEncoding = pairEncodings[0]
		}
		nAddedTokens = (*t.PostProcessor).AddedTokens(isPaired)

		if nAddedTokens > 0 {
			params := t.Trunc
			params.MaxLength = t.Trunc.MaxLength - nAddedTokens
			TruncateEncodings(encoding, *params, pairEncoding)
		} else {
			TruncateEncodings(encoding, *t.Trunc, pairEncoding)
		}
	}

	// 2. Post processing
	var finalEncoding Encoding
	if t.PostProcessor != nil {
		finalEncoding = (*t.PostProcessor).Process(encoding, pairEncoding)
	} else {
		if isPaired {
			finalEncoding = encoding
		}

		encoding.MergeWith(pairEncoding)
		finalEncoding = encoding
	}

	// 3. Padding if needed
	if t.Padding != nil {
		// We can only pad for a given size. If the Strategy is BatchLongest,
		// It will be done when we handle a batch
		var size uint
		if t.Padding.Strategy.Name == "Fixed" {
			size = t.Padding.Strategy.Value.(uint)
		} else {
			size = uint(len(finalEncoding.GetIds()))
		}

		finalEncoding.Pad(size, t.Padding.PadId, t.Padding.PadTypeId, t.Padding.PadToken, t.Padding.Direction)
	}

	return finalEncoding
}

func (t *Tokenizer) refreshAddedTokens() {
	// We need to rebuild regexp here everytime
	// because the added tokens may have changed
	var specialTokens []AddedToken
	var newTokens []string

	for k, _ := range t.SpecialTokens {
		addedTok := AddedToken{
			Content:      k,
			IsSingleWord: true,
		}
		specialTokens = append(specialTokens, addedTok)
	}

	var addedTokens []AddedToken
	for k, _ := range t.AddedTokens {
		addedTokens = append(addedTokens, k)
	}

	// merge with the one from special tokens
	addedTokens = append(addedTokens, specialTokens...)

	for _, t := range addedTokens {
		var tok string
		if t.IsSingleWord {
			var (
				firstB string // first boundary
				lastB  string // last boundary
			)
			rs := []rune(t.Content)
			firstChar := string(rs[0])
			lastChar := string(rs[len(rs)])
			isWordChar := func(char string) bool {
				m, err := regexp.MatchString(`\w`, char)
				if err != nil {
					log.Fatal(err)
				}
				return m
			}

			if isWordChar(firstChar) {
				firstB = fmt.Sprintf("%v", `\b`) // NOTE: back tick for raw string
			} else {
				firstB = ""
			}

			if isWordChar(lastChar) {
				lastB = fmt.Sprintf("%v", `\b`)
			} else {
				lastB = ""
			}

			// Escape all regular expression metacharacters
			escapeTok := regexp.QuoteMeta(t.Content)
			tok = fmt.Sprintf("%v%v%v", firstB, escapeTok, lastB)
		} else {
			tok = regexp.QuoteMeta(t.Content)
		}

		newTokens = append(newTokens, tok)
	}

	if len(newTokens) == 0 {
		t.SplitRe = nil
	}

	re := strings.Join(newTokens, "|")
	t.SplitRe = regexp.MustCompile(re)
}
