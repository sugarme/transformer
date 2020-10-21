package squad

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/sugarme/tokenizer/util"
)

type Squad2 struct {
	Version string        `json:"version"`
	Data    []SquadV2Data `json:"data"`
}

type SquadV2Data struct {
	Title      string      `json:"title"`
	Paragraphs []Paragraph `json:"paragraphs"`
}

type Paragraph struct {
	QAs     []QA   `json:"qas"`
	Context string `json:"context"`
}

type QA struct {
	Question         string   `json:"question"`
	Id               string   `json:"id"`
	Answers          []Answer `json:"answers"`
	IsImposible      bool     `json:"is_impossible"`
	PlausibleAnswers []Answer `json:"plausible_answers"`
}

type Answer struct {
	Text        string `json:"text"`
	AnswerStart int    `json:"answer_start"`
}

// Load loads SQUAD v2.0 data from file.
//
// Param
// - datasetNameOpt: specify either "train" or "dev" dataset. Default="train"
func LoadV2(datasetNameOpt ...string) []Example {
	util.CdToThis()
	datasetName := "train"
	if len(datasetNameOpt) > 0 {
		datasetName = datasetNameOpt[0]
	}

	var (
		jsonFile *os.File
		err      error
	)

	if datasetName == "train" {
		jsonFile, err = os.Open("train-v2.0.json")
	} else if datasetName == "dev" {
		jsonFile, err = os.Open("dev-v2.0.json")
	} else {
		log.Fatalf("Invalid datasetNameOpt: '%v'\n", datasetNameOpt[0])
	}
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var squad Squad2
	json.Unmarshal([]byte(byteValue), &squad)

	examples := createSquad2Examples(&squad)

	return examples
}

func createSquad2Examples(squad2 *Squad2) []Example {
	var examples []Example

	for _, doc := range squad2.Data {
		title := doc.Title
		for _, p := range doc.Paragraphs {
			context := p.Context
			for _, qa := range p.QAs {
				id := qa.Id
				question := qa.Question
				answers := qa.Answers
				// plausibleAnswers := qa.PlausibleAnswers
				isImpossible := qa.IsImposible

				for _, a := range answers {
					answerText := a.Text
					answerStart := a.AnswerStart

					newEx := NewExample(id, question, context, answerText, answerStart, title, isImpossible)
					examples = append(examples, *newEx)
				}
			}
		}
	}

	return examples
}
