package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/martinsuchenak/scriptling-llm-lib"
	"github.com/paularlott/scriptling"
	"github.com/paularlott/scriptling/object"
	"github.com/paularlott/scriptling/stdlib"
)

func main() {
	var (
		lint     bool
		eval     string
		showHelp bool
	)

	flag.BoolVar(&lint, "lint", false, "Check script for syntax errors without executing")
	flag.StringVar(&eval, "eval", "", "Evaluate a Scriptling expression")
	flag.BoolVar(&showHelp, "help", false, "Show usage information")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <script.py> [args...]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Scriptling with LLM inference primitives.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nAvailable functions: import llm\n")
		fmt.Fprintf(os.Stderr, "  argmax, argmin, topk, clip,\n")
		fmt.Fprintf(os.Stderr, "  sigmoid, relu, gelu, silu,\n")
		fmt.Fprintf(os.Stderr, "  vec_add, vec_sub, vec_mul, vec_scale, vec_apply,\n")
		fmt.Fprintf(os.Stderr, "  rms_norm, rope, silu_gate, attention, linear, linear_row,\n")
		fmt.Fprintf(os.Stderr, "  top_k, dequantize_q8,\n")
		fmt.Fprintf(os.Stderr, "  concat_rows, slice_rows, flatten\n")
	}
	flag.Parse()

	if showHelp {
		flag.Usage()
		os.Exit(0)
	}

	p := scriptling.New()
	p.EnableOutputCapture()
	stdlib.RegisterAll(p)
	p.RegisterLibrary(scriptlingllmlib.Library)

	if eval != "" {
		result, err := p.Eval(eval)
		printOutput(p)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		if result != nil {
			if _, isNull := result.(*object.Null); !isNull {
				fmt.Println(result.Inspect())
			}
		}
		return
	}

	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "Error: no script file specified\n\n")
		flag.Usage()
		os.Exit(2)
	}

	filename := args[0]
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: file not found: %s\n", filename)
		os.Exit(2)
	}

	if lint {
		_, err := p.EvalFile(filename)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Lint error: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "OK: %s\n", filename)
		return
	}

	_, err := p.EvalFile(filename)
	printOutput(p)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func printOutput(p *scriptling.Scriptling) {
	output := p.GetOutput()
	if output != "" {
		fmt.Print(output)
	}
}
