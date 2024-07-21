#!/usr/bin/env bash

export MY_UID="$(id -u)"
export MY_GID="$(id -g)"

sudo docker compose up