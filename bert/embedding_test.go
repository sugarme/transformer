package bert_test

import (
	// "encoding/json"
	// "fmt"
	// "io/ioutil"
	// "os"
	"reflect"
	"runtime"
	"testing"
)

/* func TestBert_Embedding(t *testing.T) {
 *
 *   // var path nn.Path
 *   //
 *   // config := bert.BertConfig{}
 *   //
 *   // em, err := bert.NewBertEmbedding(path, config)
 *   // if err != nil {
 *   // t.Error(err)
 *   // }
 *
 *   vs := nn.NewVarStore(nn.CPU)
 *   var config bert.BertConfig
 *   configFile, err := os.Open("../../example/testdata/bert/config.json")
 *   if err != nil {
 *     t.Error(err)
 *   }
 *   byteVal, err := ioutil.ReadAll(configFile)
 *   if err != nil {
 *     t.Error(err)
 *   }
 *   json.Unmarshal(byteVal, &config)
 *
 *   var path nn.Path
 *   path.VarStore = vs
 *   path.Var("test", []int{2}, nn.InitFloat64(1.2))
 *
 *   em, err := bert.NewBertEmbedding(path, config)
 *   if err != nil {
 *     t.Error(err)
 *   }
 *
 *   fmt.Println(path.Path)
 *
 *   t.Error(em.WordEmbeddings.Config)
 *   t.Error(em.WordEmbeddings.Ws.Data())
 *   t.Error(em.PositionEmbeddings.Config)
 *   t.Error(em.TokenTypeEmbeddings.Config)
 *
 * } */

func TestSlice1(t *testing.T) {
	defer runtime.GC()

	g := G.NewGraph()
	// x := NewTensor(g, Float64, 2, WithShape(2, 3), WithInit(RangedFrom(0)))
	// x := NewTensor(g, Float64, 2, WithShape(2, 3), WithInit(Zeroes()))
	// x := NewTensor(g, Float64, 2, WithShape(2, 3), WithInit(Ones()))
	x := G.NewTensor(g, G.Float64, 2, G.WithShape(2, 3), G.WithInit(G.ValuesOf(1.1)))
	sliced, err := G.Slice(x, G.S(0))
	if err != nil {
		t.Error(err)
	}

	want := x.Value().Data()
	got := sliced.Value()

	t.Logf("tensor:\n%v\n", x.Value())

	t.Errorf("sliced:\n%v\n", sliced.Value())

	if reflect.DeepEqual(want, got) {
		t.Errorf("sliced not equal. \n")
	}

}
