N = 5
ans = [0 for i in range(N)]
used = [False for i in range(N)]


def bt(pos):
	for i in range(N):
		if not used[i]:
			ans[pos] = i
			used[i] = True

			if pos == N:
				print(*ans)
			else:
				bt(pos+1)
			
			used[i] = False

	
