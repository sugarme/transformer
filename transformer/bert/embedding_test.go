package bert_test

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"github.com/sugarme/sermo/transformer/bert"
	"github.com/sugarme/sermo/util/nn"
)

func TestBert_Embedding(t *testing.T) {

	// var path nn.Path
	//
	// config := bert.BertConfig{}
	//
	// em, err := bert.NewBertEmbedding(path, config)
	// if err != nil {
	// t.Error(err)
	// }

	vs := nn.NewVarStore(nn.CPU)
	var config bert.BertConfig
	configFile, err := os.Open("../../example/testdata/bert/config.json")
	if err != nil {
		t.Error(err)
	}
	byteVal, err := ioutil.ReadAll(configFile)
	if err != nil {
		t.Error(err)
	}
	json.Unmarshal(byteVal, &config)

	var path nn.Path
	path.VarStore = vs
	path.Var("test", []int{2}, nn.InitFloat64(1.2))

	em, err := bert.NewBertEmbedding(path, config)
	if err != nil {
		t.Error(err)
	}

	fmt.Println(path.Path)

	t.Error(em.WordEmbeddings.Config)
	t.Error(em.WordEmbeddings.Ws.Data())
	t.Error(em.PositionEmbeddings.Config)
	t.Error(em.TokenTypeEmbeddings.Config)

}
