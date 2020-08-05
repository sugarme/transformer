package common

/*
 * import (
 *   "encoding/json"
 *   "io/ioutil"
 *   "log"
 *   "os"
 *   "path/filepath"
 *
 *   "github.com/sugarme/transformer/bert"
 * )
 *
 * type Config interface{}
 *
 * // ConfigFromFile loads a Config object from json file
 * // The format should be the same as in [Transformers library](https://github.com/huggingface/transformers)
 * // It will be panic if any non-optional parameters expected by model are missing.
 * func ConfigFromFile(filename string, modelType string) (retVal Config) {
 *
 *   filePath, err := filepath.Abs(filename)
 *   if err != nil {
 *     log.Fatal(err)
 *   }
 *
 *   f, err := os.Open(filePath)
 *   if err != nil {
 *     log.Fatal(err)
 *   }
 *   defer f.Close()
 *
 *   buff, err := ioutil.ReadAll(f)
 *   if err != nil {
 *     log.Fatal(err)
 *   }
 *
 *   switch modelType {
 *   case "BERT":
 *     var config bert.BertConfig
 *     err = json.Unmarshal(buff, &config)
 *     if err != nil {
 *       log.Fatalf("Could not parse configuration to BertConfiguration.\n")
 *     }
 *     return config
 *   default:
 *     err = json.Unmarshal(buff, &retVal)
 *     if err != nil {
 *       log.Fatal(err)
 *     }
 *     return retVal
 *   }
 * } */
