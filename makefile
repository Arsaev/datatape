BINARY = datatape

.PHONY: all build test clean

all: build test

build: $(BINARY)

$(BINARY): *.go
	@echo "Building..."
	go build -o $(BINARY) .

test: build input.txt
	@echo "Testing encode..."
	./$(BINARY) -encode -input input.txt -output output.wav
	@echo "Testing decode..."
	./$(BINARY) -decode -input output.wav -output recovered.txt
	@echo "Comparing files..."
	@if cmp -s input.txt recovered.txt; then \
		echo "✓ SUCCESS: Files match!"; \
	else \
		echo "✗ FAILURE: Files differ"; \
		diff input.txt recovered.txt; \
	fi
	@echo "Original: $$(wc -c < input.txt) bytes"
	@echo "Audio:    $$(wc -c < output.wav) bytes" 
	@echo "Recovered: $$(wc -c < recovered.txt) bytes"

clean:
	rm -f $(BINARY) input.txt output.wav recovered.txt