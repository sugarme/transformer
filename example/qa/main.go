package main

import (
	"fmt"
	"log"
	// "strings"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
	"github.com/sugarme/tokenizer/pretrained"
	"github.com/sugarme/transformer/bert"
)

func main() {
	// Config
	config, err := bert.ConfigFromFile("../../data/bert/config-qa.json")
	if err != nil {
		log.Fatal(err)
	}

	// Model
	device := gotch.CPU
	// device := gotch.NewCuda().CudaIfAvailable()
	vs := nn.NewVarStore(device)

	model := bert.NewBertForQuestionAnswering(vs.Root(), config)
	err = vs.Load("../../data/bert/bert-qa.ot")
	if err != nil {
		log.Fatalf("Load model weight error: \n%v", err)
	}

	fmt.Printf("Varstore weights have been loaded\n")
	fmt.Printf("Num of variables: %v\n", len(vs.Variables()))

	// NOTE. BERT finetuned for question answering used different vocab file.
	tk := pretrained.BertLargeCasedWholeWordMaskingSquad()
	// tk := getBert()

	// question := "How many parameters does BERT-large have?"
	// context := "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."
	// context := "The US has passed the peak on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world."
	// question := "What was President Donald Trump's prediction?"

	// context := "Remi is living in Australia."
	// question := "Where does Remi live?"

	// context := "New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean. It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of 4.9 million. New Zealand's capital city is Wellington, and its most populous city is Auckland."
	// question := "How many people live in New Zealand?"

	context := `Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.
Most people who fall sick with COVID-19 will experience mild to moderate symptoms and recover without special treatment.
The virus that causes COVID-19 is mainly transmitted through droplets generated when an infected person coughs, sneezes, or exhales. These droplets are too heavy to hang in the air, and quickly fall on floors or surfaces.
You can be infected by breathing in the virus if you are within close proximity of someone who has COVID-19, or by touching a contaminated surface and then your eyes, nose or mouth.
The most common symptoms of COVID-19 are

Fever
Dry cough
Fatigue
Other symptoms that are less common and may affect some patients include:

Loss of taste or smell,
Nasal congestion,
Conjunctivitis (also known as red eyes)
Sore throat,
Headache,
Muscle or joint pain,
Different types of skin rash,
Nausea or vomiting,
Diarrhea,
Chills or dizziness.
Symptoms are usually mild. Some people become infected but only have very mild symptoms or none at all.
Symptoms of severe COVID‐19 disease include:
Shortness of breath,
Loss of appetite,
Confusion,
Persistent pain or pressure in the chest,
High temperature (above 38 °C).
Other less common symptoms are:
Irritability,
Confusion,
Reduced consciousness (sometimes associated with seizures),
Anxiety,
Depression,
Sleep disorders,
More severe and rare neurological complications such as strokes, brain inflammation, delirium and nerve damage.
People of all ages who experience fever and/or cough associated with difficulty breathing or shortness of breath, chest pain or pressure, or loss of speech or movement should seek medical care immediately. If possible, call your health care provider, hotline or health facility first, so you can be directed to the right clinic.
`
	// question := "How covid 19 spreads?"
	// question := "What are symptoms of covid 19?"
	// question := "What are severe signs of covid 19?"
	// question := "What are uncommon signs of covid 19?" // NOTE. Will be panic due to `answerStart` > `answerEnd`
	question := "What is covid 19?"

	encoding, err := tk.EncodePair(question, context, true)
	if err != nil {
		log.Fatal(err)
	}

	var batchSize int64 = 1
	var seqLen int64 = int64(len(encoding.Ids))

	inputTensor := ts.MustOfSlice(toInt64(encoding.Ids)).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true)
	tokenTypeIds := ts.MustOfSlice(toInt64(encoding.TypeIds)).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true)
	// positionIds := ts.MustOfSlice(toInt64(encoding.Words)).MustTo(device, true).MustView([]int64{batchSize, seqLen}, true).MustExpand([]int64{batchSize, seqLen}, true, true)

	// positionIds := ts.MustArange(ts.IntScalar(seqLen), gotch.Int64, device).MustExpand([]int64{batchSize, seqLen}, true, true)
	// mask := ts.MustOnes([]int64{batchSize, seqLen}, gotch.Int64, device)

	var startScores, endScores *ts.Tensor
	ts.NoGrad(func() {
		startScores, endScores, _, _, err = model.ForwardT(inputTensor, ts.None, tokenTypeIds, ts.None, ts.None, false)
		if err != nil {
			log.Fatal(err)
		}
	})

	answerStart := startScores.MustGet(0).MustArgmax([]int64{0}, false, false).Int64Values()[0]
	answerEnd := endScores.MustGet(0).MustArgmax([]int64{0}, false, false).Int64Values()[0]

	fmt.Printf("answer start: '%v'\n", answerStart)
	fmt.Printf("answer end: '%v'\n", answerEnd)

	// NOTE. To prevent picking `answerStart` > `answerEnd`, we need to find
	// the highest total score for which end >= start
	// TODO. implement it (pytorch topK) to find topK scores with pair(start, end)
	// then pick the highest score where start < end
	// answerTokens := encoding.Tokens[answerStart : answerEnd+1]
	answerIds := encoding.Ids[answerStart : answerEnd+1]
	if int(answerEnd) >= len(encoding.Tokens) {
		// answerTokens = encoding.Tokens[answerStart:answerEnd]
		answerIds = encoding.Ids[answerStart:answerEnd]
	}

	answerStr := tk.Decode(answerIds, false)

	// fmt.Printf("context: '%v'\n", strings.Join(encoding.Tokens, " "))
	// fmt.Printf("Answer: '%v'\n", strings.Join(answerTokens, " "))
	fmt.Printf("Context: %v\n", context)
	fmt.Printf("Question: '%v'\n", question)
	fmt.Printf("Answer: '%v'\n", answerStr)

}

func toInt64(data []int) []int64 {
	var data64 []int64
	for _, v := range data {
		data64 = append(data64, int64(v))
	}

	return data64
}

func filterPosition(data []int) []int {
	var filterData []int
	for _, v := range data {
		if v != -1 {
			filterData = append(filterData, v)
		}
	}
	return filterData
}
