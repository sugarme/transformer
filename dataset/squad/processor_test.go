package squad_test

import (
	// "reflect"
	"fmt"
	"testing"

	"github.com/sugarme/tokenizer/pretrained"
	"github.com/sugarme/transformer/dataset/squad"
)

func TestConvertExampleToFeatures(t *testing.T) {

	tk := pretrained.BertLargeCasedWholeWordMaskingSquad()

	question := "Where is remi living?"
	context := "Remi has been living in Australia since he was born."
	answer := "to update"
	startPositionChar := 0
	answers := []squad.Answer{}
	title := "qa testing"

	var example *squad.Example = squad.NewExample("1", question, context, answer, startPositionChar, title, answers, false)

	features := squad.ConvertExamplesToFeatures([]squad.Example{*example}, tk, "bert", 384, 128, 64, "[SEP]", "[PAD]", 0, false)

	fmt.Printf("features: %+v\n", features[0])

	t.Errorf("Stops\n")

}
