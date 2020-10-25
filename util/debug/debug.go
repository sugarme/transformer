package debug

import (
	"fmt"
)

var currentUsed uint64

func init() {
	var si *SI
	si = CPUInfo()
	fmt.Printf("Total RAM (MB):\t %8.2f\n", float64(si.TotalRam)/1024)
	fmt.Printf("Used RAM (MB):\t %8.2f\n", float64(si.TotalRam-si.FreeRam)/1024)
	currentUsed = si.TotalRam - si.FreeRam
}

// UsedRAM returns used-up RAM in MiB since last check.
func UsedRAM() {
	var si *SI
	si = CPUInfo()
	used := (float64(si.TotalRam-si.FreeRam) - float64(currentUsed)) / 1024
	currentUsed = si.TotalRam - si.FreeRam

	fmt.Printf("Used RAM: [%8.2f MiB]\n", used)
}

func UsedGPU() {
	GPUInfo()
}
