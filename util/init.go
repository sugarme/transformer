package util

import (
	"fmt"
	"log"
	"os"
)

var (
	CachedDir         string = "NOT_SETTING"
	transformerEnvKey string = "GO_TRANSFORMER"
)

func init() {
	// default path: {$HOME}/.cache/transfomer
	homeDir := os.Getenv("HOME")
	CachedDir = fmt.Sprintf("%s/.cache/transformer", homeDir)

	initEnv()

	log.Printf("INFO: CachedDir=%q\n", CachedDir)
}

func initEnv() {
	val := os.Getenv(transformerEnvKey)
	if val != "" {
		CachedDir = val
	}

	if _, err := os.Stat(CachedDir); os.IsNotExist(err) {
		if err := os.MkdirAll(CachedDir, 0755); err != nil {
			log.Fatal(err)
		}
	}
}
