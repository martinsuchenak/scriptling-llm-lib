package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/martinsuchenak/scriptling-llm-lib"
	"github.com/paularlott/scriptling"
	"github.com/paularlott/scriptling/extlibs"
	"github.com/paularlott/scriptling/libloader"
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
		fmt.Fprintf(os.Stderr, "Scriptling runtime with LLM inference support.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nAvailable libraries:\n")
		fmt.Fprintf(os.Stderr, "  llm   — LLM inference primitives and generate()\n")
		fmt.Fprintf(os.Stderr, "  math  — standard math operations\n")
		fmt.Fprintf(os.Stderr, "  os    — file system operations\n")
		fmt.Fprintf(os.Stderr, "  fs    — binary file I/O\n")
		fmt.Fprintf(os.Stderr, "  sys   — system/argv access\n")
	}
	flag.Parse()

	if showHelp {
		flag.Usage()
		os.Exit(0)
	}

	p := scriptling.New()
	stdlib.RegisterAll(p)
	p.RegisterLibrary(scriptlingllmlib.Library)

	wd, _ := os.Getwd()
	extlibs.RegisterOSLibrary(p, []string{wd})
	extlibs.RegisterFSLibrary(p, []string{wd})
	extlibs.RegisterRuntimeLibrary(p)

	if eval != "" {
		result, err := p.Eval(eval)
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
	scriptArgs := append([]string{filename}, args[1:]...)
	extlibs.RegisterSysLibrary(p, scriptArgs, os.Stdin)
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: file not found: %s\n", filename)
		os.Exit(2)
	}

	scriptDir, _ := filepath.Abs(filepath.Dir(filename))
	p.SetLibraryLoader(libloader.NewFilesystem(scriptDir))

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
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
