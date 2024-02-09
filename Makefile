.PHONY: clean

CFLAGS  := -Wall  -g
LD      := gcc
LDLIBS  := ${LDLIBS} -lrdmacm -libverbs -lpthread

APPS    := rdma-client rdma-server

all: ${APPS}

rdma-client: client-utils.o rdma-client.o
	${LD} -o $@ $^ ${LDLIBS}

rdma-server: server-utils.o rdma-server.o
	${LD} -o $@ $^ ${LDLIBS}

clean:
	rm -f *.o ${APPS}

