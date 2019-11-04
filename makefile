CC = clang++
LLDB = -g
BOOST = -lboost_serialization
PYTHON = -lboost_python37
INCLUDE = -I/usr/local/lib `python3.7-config --include`
LDFLAGS = `python3.7-config --ldflags`

hpylm:
	$(CC) -O3 -DPIC -shared -fPIC -o model.so src/model.cpp $(INCLUDE) $(LDFLAGS) $(PYTHON) $(BOOST)

test:
	$(CC) -O3 -DPIC -shared -fPIC -o test src/test.cpp $(LLDB) $(INCLUDE) $(LDFLAGS) $(PYTHON) $(BOOST)

clean:
	rm -f model.so test

.PHONY: clean