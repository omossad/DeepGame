#!/bin/bash
./../vmaf/libvmaf/build/tools/vmafossexec yuv420p "$3" "$4" "$1" "$2enc.yuv"  ../vmaf/model/vmaf_v0.6.1.pkl --log "$2vmaf_output.xml"  --thread 8
