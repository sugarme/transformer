package debug

import (
	"fmt"
	"log"
)

var machine Machine

type Machine struct {
	CPUMemTotal uint64
	CPUMemUsed  uint64

	GPUs        int // number of GPUs
	GPUMemTotal uint64
	GPUMemUsed  uint64
}

func init() {
	var si *SI
	si = CPUInfo()

	machine.CPUMemTotal = si.TotalRam
	machine.CPUMemUsed = si.TotalRam - si.FreeRam
	machine.GPUs = CountGPUs()
	gpuTotal, err := GPUMemory("total")
	if err != nil {
		log.Println(err.Error())
		machine.GPUMemTotal = 0
	} else {
		machine.GPUMemTotal = gpuTotal
	}
	gpuUsed, err := GPUMemory("used")
	if err != nil {
		log.Println(err.Error())
		machine.GPUMemUsed = 0
	} else {
		machine.GPUMemUsed = gpuUsed
	}

	fmt.Printf("RAM Total:\t\t %8.0f(MiB) \tUsed:\t %8.0f(MiB)\n", float64(si.TotalRam)/1024, float64(si.TotalRam-si.FreeRam)/1024)
	fmt.Printf("GPU memory Total:\t %8.0f(MiB) \tUsed:\t %8.0f(MiB)\n", float64(machine.GPUMemTotal), float64(machine.GPUMemUsed))
}

// UsedRAM returns used-up RAM in MiB since last check.
func UsedCPUMem() float64 {
	var si *SI
	si = CPUInfo()

	// update
	prev := machine.CPUMemUsed
	curr := si.TotalRam - si.FreeRam
	machine.CPUMemUsed = curr

	// to MiB
	return float64(curr-prev) / 1024
}

// UsedGPUMem returns used-up GPU memory in MiB since last check.
// If error occurs, it will log to console and return -1.
func UsedGPUMem() float64 {
	prev := machine.GPUMemUsed
	curr, err := GPUMemory("used")
	if err != nil {
		log.Printf(err.Error())
		return -1
	}
	machine.GPUMemUsed = curr

	return float64(curr - prev)
}
