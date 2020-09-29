package util

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"strings"
)

const (
	WeightName = "model.gt" // atm, just use extension name `.ot`
	ConfigName = "config.json"

	S3BucketPrefix          = "https://s3.amazonaws.com/models.huggingface.co/bert"
	CloudfrontDistribPrefix = "https://cdn.huggingface.co"
)

var (
	DefaultCachePath           string
	PytorchPretrainedBertCache string
	PytorchTransformersCache   string
	TransformersCache          string
)

func init() {
	// Default values
	userHome, err := os.UserHomeDir()
	if err != nil {
		log.Fatal(err)
	}
	DefaultCachePath = fmt.Sprintf("%v/.cache/transformers", userHome)
	if _, err := os.Stat(DefaultCachePath); os.IsNotExist(err) {
		if err := os.MkdirAll(DefaultCachePath, 0755); err != nil {
			log.Fatal(err)
		}
	}

	PytorchTransformersCache = DefaultCachePath
	TransformersCache = PytorchTransformersCache

	if val, ok := os.LookupEnv("PytorchPretrainedBertCache"); ok {
		PytorchTransformersCache = val
	}

	if val, ok := os.LookupEnv("PytorchTransformersCache"); ok {
		TransformersCache = val
	}
}

// CachedPath resolves and caches data based on input string, then returns fullpath to the cached data.
//
// Parameters:
// - `urlOrFilename`: can be either `URL` to remote file or fullpath a local file.
//
// CachedPath does several things consequently:
// 1. Resolves input string to a  fullpath cached filename candidate.
// 2. Check it at `CachePath`, if exists, then return the candidate. If not
// 3. Retrieves and Caches data to `CachePath` and returns path to cached data
func CachedPath(urlOrFilename string) (resolvedPath string, err error) {

	// 1. Resolves to "candidate" filename at `CachePath`
	filename := path.Base(urlOrFilename)
	cachedFileCandidate := fmt.Sprintf("%s/%v", TransformersCache, filename)

	// 1. Cached candidate exists
	// if _, err := os.Stat(cachedFileCandidate); os.IsExist(err) {
	if _, err := os.Stat(cachedFileCandidate); err == nil {
		return cachedFileCandidate, nil
	}

	// 2. If valid fullpath to local file, caches it and return cached filename
	if _, err := os.Stat(urlOrFilename); err == nil {
		err := copyFile(urlOrFilename, cachedFileCandidate)
		if err != nil {
			return "", err
		}
		return cachedFileCandidate, nil
	}

	// 3. If a valid URL, download it to `CachePath`
	if isValidURL(urlOrFilename) {
		if _, err := http.Get(urlOrFilename); err == nil {
			err := downloadFile(urlOrFilename, cachedFileCandidate)
			if err != nil {
				return "", err
			}
		} else {
			err = fmt.Errorf("Unable to parse '%v' as a URL or as a local path.\n", urlOrFilename)
			return "", err
		}
	}

	// Not resolves
	err = fmt.Errorf("Unable to parse '%v' as a URL or as a local path.\n", urlOrFilename)
	return "", err
}

func isValidURL(url string) bool {

	// TODO: implement
	return false
}

// downloadFile downloads file from URL and stores it in local filepath.
// It writes to the destination file as it downloads it, without loading
// the entire file into memory. An `io.TeeReader` is passed into Copy()
// to report progress on the download.
func downloadFile(url string, filepath string) error {

	// Create the file with .tmp extension, so that we won't overwrite a
	// file until it's downloaded fully
	out, err := os.Create(filepath + ".tmp")
	if err != nil {
		return err
	}
	defer out.Close()

	// Get the data
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Create our bytes counter and pass it to be used alongside our writer
	counter := &writeCounter{}
	_, err = io.Copy(out, io.TeeReader(resp.Body, counter))
	if err != nil {
		return err
	}

	// The progress use the same line so print a new line once it's finished downloading
	fmt.Println()

	// Rename the tmp file back to the original file
	err = os.Rename(filepath+".tmp", filepath)
	if err != nil {
		return err
	}

	return nil
}

// writeCounter counts the number of bytes written to it. By implementing the Write method,
// it is of the io.Writer interface and we can pass this into io.TeeReader()
// Every write to this writer, will print the progress of the file write.
type writeCounter struct {
	Total uint64
}

func (wc *writeCounter) Write(p []byte) (int, error) {
	n := len(p)
	wc.Total += uint64(n)
	wc.printProgress()
	return n, nil
}

// PrintProgress prints the progress of a file write
func (wc writeCounter) printProgress() {
	// Clear the line by using a character return to go back to the start and remove
	// the remaining characters by filling it with spaces
	fmt.Printf("\r%s", strings.Repeat(" ", 50))

	// Return again and print current status of download
	fmt.Printf("\rDownloading... %s complete", byteCountIEC(wc.Total))
}

// byteCountIEC converts bytes to human-readable string in binary (IEC) format.
func byteCountIEC(b uint64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := uint64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB",
		float64(b)/float64(div), "KMGTPE"[exp])
}

func copyFile(src, dst string) error {
	sourceFileStat, err := os.Stat(src)
	if err != nil {
		return err
	}

	if !sourceFileStat.Mode().IsRegular() {
		return fmt.Errorf("%s is not a regular file", src)
	}

	source, err := os.Open(src)
	if err != nil {
		return err
	}
	defer source.Close()

	destination, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destination.Close()
	_, err = io.Copy(destination, source)
	return err
}
