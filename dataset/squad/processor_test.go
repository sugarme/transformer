package squad_test

import (
	// "reflect"
	"fmt"
	"testing"

	"github.com/sugarme/tokenizer/pretrained"
	"github.com/sugarme/transformer/dataset/squad"
)

func TestConvertExampleToFeatures(t *testing.T) {

	tk := pretrained.BertBaseUncased()

	question := "How many parameters does BERT-large have?"
	context := "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."
	answer := "to update"
	startPositionChar := 0
	answers := []squad.Answer{}
	title := "qa testing"

	var example *squad.Example = squad.NewExample("1", question, context, answer, startPositionChar, title, answers, false)

	features := squad.ConvertExampleToFeatures(tk, *example, 512, 0, 512, true, false)

	encoding, err := tk.EncodePair(question, context, true)
	if err != nil {
		t.Error(err)
	}

	fmt.Printf("Tokens: %q\nLength: %v\n\n", encoding.Tokens, len(encoding.Tokens))
	fmt.Printf("Ids: %v\nLength: %v\n\n", encoding.Ids, len(encoding.Ids))
	fmt.Printf("Type Ids: %v\n\n", encoding.TypeIds)

	fmt.Printf("Context: %v\n\n", example.ContextText)
	fmt.Printf("DocTokens: %q\n\n", example.DocTokens)
	fmt.Printf("features: %+v\n", features)

	t.Errorf("Stops\n")

}
