#!/bin/bash
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
#
# This code is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this code. If not, see <http://www.gnu.org/licenses/>.


for exp in box stair
do
    echo "Press [Enter] to analyze logs for experiment '$exp'"
    read
    for file in $exp/segment_*.txt
    do
        echo "$file"
        echo "================================="
        echo ""

        cat $file | grep "CGI shape" | uniq
        cat $file | grep "nb points" | uniq
        cat $file | grep "discr" | uniq

        echo ""

        echo -n "Compute CGI:       "
        cat $file \
            | grep "Compute CGI" \
            | cut -d ' ' -f 5 \
            | awk '{delta = $file - avg; avg += delta / NR; mean2 += delta * ($file - avg); } END { print avg, "+/-", sqrt(mean2 / NR); }'

        echo -n "Compute (a, b, c): "
        cat $file \
            | grep "(a, b, c)" \
            | cut -d ' ' -f 8 \
            | awk '{delta = $file - avg; avg += delta / NR; mean2 += delta * ($file - avg); } END { print avg, "+/-", sqrt(mean2 / NR); }'

        echo -n "Compute TOPP:      "
        cat $file \
            | grep "TOPP comp" \
            | cut -d ' ' -f 5 \
            | awk '{delta = $file - avg; avg += delta / NR; mean2 += delta * ($file - avg); } END { print avg, "+/-", sqrt(mean2 / NR); }'

        echo ""
    done
done
