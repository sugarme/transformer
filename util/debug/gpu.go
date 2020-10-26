package debug

import (
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"strings"
)

// CountGPUs returns number of Nvidia GPUs.
func CountGPUs() int {
	cmd := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
	stdout, err := cmd.Output()
	if err != nil {
		log.Printf("nvidia-smi cli error: %v\n", err.Error())
		return 0
	}
	n := strings.Count(string(stdout), "\n")
	return n
}

// GPUInfo prints out nvidia GPU information to console.
func GPUInfo() {
	nvidia := "nvidia-smi"
	cmd := exec.Command(nvidia)
	stdout, err := cmd.Output()

	if err != nil {
		log.Printf("nvidia-smi cli error: %v\n", err.Error())
	}

	fmt.Printf("%v\n", string(stdout))

}

// GPUMemory returns specified memory information in MiB unit
//
// `typeOpt`: type of memory info. Can be "used", "total", "free". Default="total"
func GPUMemory(typeOpt ...string) (uint64, error) {
	var memType string = "total"
	var err error
	if len(typeOpt) > 0 {
		opt0 := typeOpt[0]
		if opt0 != "total" && opt0 != "used" && opt0 != "free" {
			err = fmt.Errorf("Invalid GPU memory typeOpt: %v. 'typeOpt' can be 'total', 'used' or 'free' only.\n", opt0)
			return 0, err
		}
		memType = opt0
	}

	queryStr := fmt.Sprintf("--query-gpu=memory.%v", memType)

	nvidia := "nvidia-smi"
	cmd := exec.Command(nvidia, queryStr, "--format=csv,noheader,nounits")
	stdout, err := cmd.Output()

	if err != nil {
		err = fmt.Errorf("nvidia-smi cli error: %v\n", err.Error())
		return 0, err
	}

	used, err := strconv.Atoi(strings.TrimSuffix(string(stdout), "\n"))
	if err != nil {
		err = fmt.Errorf("nvidia-smi cli error: %v\n", err.Error())
		return 0, err
	}

	return uint64(used), nil
}
