package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"sync"
)

const mb = 1024 * 1024
const gb = 1024 * mb

func main() {
	wg := new(sync.WaitGroup)

	// word channel to send read words for processing
	wChan := make(chan ([]string))

	// dictionary of words and their frequency
	dict := make(map[string]uint32)

	// channel to signal the main thread that all the words have been
	doneChan := make(chan (bool), 1)

	// Read all incoming words from the channel and add to the dict
	go func() {
		for words := range wChan {
			for _, w := range words {
				dict[w]++
			}
		}

		fmt.Printf("Dictionary length: %v words\n", len(dict))

		for k, v := range dict {
			if v > 10 {
				fmt.Printf("Word has more than 10 times occurrences: %v\n", k)
			}
		}

		// signal the main thread all done with this goroutine
		doneChan <- true
	}()

	// current is the counter for bytes of the file.
	var current int64 = 0
	var limit int64 = 100 * mb

	// Setup some workers to process
	for i := 0; i < 3; i++ {
		wg.Add(1)

		go func(i int) {
			// start reading file chunk by chunk
			current = read(current, limit, "oscar.eo.txt", wChan)
			fmt.Printf("%d thread has been completed\n", i)
			wg.Done()
		}(i)
	}

	// wait for all goroutines to complete
	wg.Wait()
	close(wChan)

	// wait for dictionary to process all words then close
	<-doneChan
	close(doneChan)

}

func read(offset int64, limit int64, filename string, channel chan ([]string)) (current int64) {
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// move the pointer of the file to the start of designated chunk
	file.Seek(offset, 0) // 0 means relative to the origin of file

	// reader := bufio.NewReader(file)
	scanner := bufio.NewScanner(file)
	buf := make([]byte, 0, 1*gb)
	scanner.Buffer(buf, 10*gb)

	var cummulativeSize int64

	for scanner.Scan() {
		// Stop if read size has exceed the chunk size
		cummulativeSize += int64(len(scanner.Bytes()))
		if cummulativeSize > limit {
			break
		}

		line := scanner.Text()

		words := strings.Split(line, " ")

		channel <- words
	}

	return cummulativeSize

}
