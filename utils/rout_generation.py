rts = {}
row_num = 3
col_num = 3
for i in range(row_num):
    route_arr_bot = []
    route_arr_top = []
    for j in range(col_num + 1):
        route_arr_bot += ["bot" + str(i) + '_' + str(j)]
        route_arr_top += ["top" + str(i) + '_' + str(col_num - j)]
    rts.update({"bot" + str(i) + '_' + '0': route_arr_bot})
    rts.update({"top" + str(i) + '_' + str(col_num): route_arr_top})

for i in range(col_num):
    route_arr_left = []
    route_arr_right = []
    for j in range(row_num + 1):
        route_arr_right += ["right" + str(j) + '_' + str(i)]
        route_arr_left += ["left" + str(row_num - j) + '_' + str(i)]
    rts.update({"left" + str(row_num) + '_' + str(i): route_arr_left})
    rts.update({"right" + '0' + '_' + str(i): route_arr_right})
print (rts)
print (len(rts))