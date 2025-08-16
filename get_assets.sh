#!/usr/bin/env bash

echo "[STATUS] Getting analog_FM_France from IQEngine.org..."
wget https://www.iqengine.org/api/datasources/gnuradio/iqengine/analog_FM_France.sigmf-data
wget https://www.iqengine.org/api/datasources/gnuradio/iqengine/analog_FM_France.sigmf-meta
echo "[STATUS] Getting NOAA-APT wav file (post WFM demod)..."
wget https://project.markroland.com/weather-satellite-imaging/N18_4827.zip
