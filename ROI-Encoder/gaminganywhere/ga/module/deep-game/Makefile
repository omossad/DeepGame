
include ../Makefile.common

ifeq ($(OS), MSYS)
LDFLAGS	+= ../../core/libga.dll $(AVCLD)
endif

OBJS	= dg.o
TARGET	= dg.$(EXT)

include ../Makefile.build

