package squad

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
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

func Load() {

	// jsonFile, err := os.Open("./dev-v2.0.json")
	jsonFile, err := os.Open("./train-v2.0.json")
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	// var result map[string]interface{}
	var squad Squad2

	json.Unmarshal([]byte(byteValue), &squad)

	fmt.Printf("number of samples: %v\n", len(squad.Data))
	fmt.Printf("Sample 0===================:\n")
	s := squad.Data[0]
	fmt.Printf("Title : %+v\n", s.Title)
	fmt.Printf("Paragraph 0: ==================\n")
	p := s.Paragraphs[0]
	fmt.Printf("Context : %+v\n", p.Context)
	for i, qa := range p.QAs {
		fmt.Printf("%v-Id:%v- Question: '%v'\nAnswers: %+v\nPlausibleAnswers: %+v\nIsImpossible: %v\n", i, qa.Id, qa.Question, qa.Answers, qa.PlausibleAnswers, qa.IsImposible)
	}

	answer1 := []rune(p.Context)[269 : len([]rune("in the late 1990s"))+269]
	fmt.Printf("answer1: '%v'\n", string(answer1))
}
