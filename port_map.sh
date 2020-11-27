#!/usr/bin/env bash
ssh -f wangshengguang@remotehost -N -L 7474:localhost:7474
ssh -f wangshengguang@remotehost -N -L 7687:localhost:7687

