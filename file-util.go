package transformer

import (
	"fmt"
	"io"
	"log"
	"path"
	"strings"

	// "path/filepath"
	"net/http"
	"os"
)

// cachedPath resolves path given an input either a URL or a local path.
// If it is a URL, download file and cache it, then return path to the cached file.
// If it's already a local path, make sure the file exists and then return the path.
/* func cachedPath(urlOrFilename string, cacheDir string, forceDownload bool, proxies []interface{}, resumeDownload bool, userAgent []interface{}, extractCompressedFile bool, localFilesOnly bool) (resolvedPath string, ok bool) {
 *
 *   // TODO: implement this
 *   return
 * } */

const (
	WeightName = "model.gt"
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

func cachedPath(urlOrFilename string, opts map[string]interface{}) (resolvedPath string, err error) {

	// Default values
	params := map[string]interface{}{
		"cacheDir":              "",
		"forcedDownload":        false,
		"proxies":               nil,
		"resumeDownload":        false,
		"userAgent":             nil,
		"extractCompressedFile": false,
		"forceExtract":          false,
		"localFilesOnly":        false,
	}

	// Update with custom values
	for k, v := range opts {
		if _, ok := params[k]; ok {
			params[k] = v
		}
	}

	var (
		outputPath string
	)
	if val, _ := params["cacheDir"]; val == "" {
		params["cacheDir"] = TransformersCache
	}

	// 1. If filename, just return it if existing, otherwise throw error
	if _, err := os.Stat(urlOrFilename); err == nil {
		return urlOrFilename, nil
	} else if _, err := http.Get(urlOrFilename); err == nil {
		// URL, so get it from the cache (downloading if necessary)
		outputPath = getFromCache(urlOrFilename, params)
	} else {
		err = fmt.Errorf("Unable to parse '%v' as a URL or as a local path.\n", urlOrFilename)
		return "", err
	}

	if !params["extractCompressedFile"].(bool) {
		return outputPath, nil
	}

	// TODO: do extract `.zip` or `.tar` file
	return outputPath, nil
}

func getFromCache(url string, opts map[string]interface{}) string {

	// Default values
	params := map[string]interface{}{
		"cacheDir":              "",
		"forcedDownload":        false,
		"proxies":               nil,
		"resumeDownload":        false,
		"userAgent":             nil,
		"extractCompressedFile": false,
		"forceExtract":          false,
		"localFilesOnly":        false,
	}

	// Update with custom values
	for k, v := range opts {
		if _, ok := params[k]; ok {
			params[k] = v
		}
	}

	if val, _ := params["cacheDir"]; val == "" {
		params["cacheDir"] = TransformersCache
	}

	if _, err := os.Stat(params["cacheDir"].(string)); os.IsNotExist(err) {
		// Does not exist. Create path
		if err := os.Mkdir(params["cacheDir"].(string), 0755); os.IsExist(err) {
			log.Fatal(err)
		}
	}

	filename := path.Base(url)
	// get cache path to put the file
	cachePath := fmt.Sprintf("%s/%s", params["cacheDir"].(string), filename)

	// If cachePath exists (meaning its cached) or `localFilesOnly`== true
	// just return cachePath
	localFilesOnly, _ := params["localFilesOnly"].(bool)
	var isCached bool = false
	if _, err := os.Stat(cachePath); err == nil {
		isCached = true
	}
	if localFilesOnly || isCached {
		fmt.Printf("cached file found.\n")
		return cachePath
	}

	// Otherwise, download file from URL
	err := downloadFile(url, cachePath)
	if err != nil {
		log.Fatal(err)
	}

	if params["resumeDownload"].(bool) {
		// TODO: do resume if incomplete
	}

	return cachePath
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
