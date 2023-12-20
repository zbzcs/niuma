最大连续子数组和


# 从输入台获取数组
input_array = list(map(int, input("请输入整数数组").split(',')))



if not input_array:
    print("输入数组为空")
else:
    current_sum = max_sum = input_array[0]

    for num in input_array[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    print("最大连续子数组和:", max_sum)


Java

import java.util.Scanner;

public class MaxSubarraySum {
    public static void main(String[] args) {
        // 从输入台获取数组
        Scanner scanner = new Scanner(System.in);
        System.out.print("请输入整数数组，以逗号分隔: ");
        String[] inputArrayStr = scanner.nextLine().split(",");
        
        int[] inputArray = new int[inputArrayStr.length];
        for (int i = 0; i < inputArrayStr.length; i++) {
            inputArray[i] = Integer.parseInt(inputArrayStr[i]);
        }
    
        if (inputArray.length == 0) {
            System.out.println("输入数组为空");
        } else {
            int currentSum = inputArray[0];
            int maxSum = inputArray[0];
    
            for (int i = 1; i < inputArray.length; i++) {
                int num = inputArray[i];
                currentSum = Math.max(num, currentSum + num);
                maxSum = Math.max(maxSum, currentSum);
            }
    
            System.out.println("最大连续子数组和: " + maxSum);
        }
    }
}



最长递增路径

class Solution(object):
    def longestIncreasingPath(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        m, n = len(matrix), len(matrix[0])
        lst = []
        for i in range(m):
            for j in range(n):
                lst.append((matrix[i][j], i, j))
        lst.sort()
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for num, i, j in lst:
            dp[i][j] = 1
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = i + di, j + dj
                if 0 <= r < m and 0 <= c < n:
                    if matrix[i][j] > matrix[r][c]:
                        dp[i][j] = max(dp[i][j], 1 + dp[r][c])
        return max([dp[i][j] for i in range(m) for j in range(n)])

Java
class Solution {
    int maxLen=0;
    public int longestIncreasingPath(int[][] matrix) {
        int m=matrix.length,n=matrix[0].length;
        int[][]visited=new int[m][n];//计算以每个节点开头的递增序列的长度
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                maxLen=Math.max(maxLen,dfs(matrix,i,j,m,n,visited,Integer.MIN_VALUE));
            }
        }
        return maxLen;
    }
    public int dfs(int[][]matrix,int i,int j,int m,int n,int[][]visited,int pre){
        if(i<0||i>=m||j<0||j>=n||matrix[i][j]<=pre) return 0;
        if(visited[i][j]>0) return visited[i][j];//如果之前已经计算过，直接返回即可
        int l= dfs(matrix,i-1,j,m,n,visited,matrix[i][j]);
        int r= dfs(matrix,i+1,j,m,n,visited,matrix[i][j]);
        int up=dfs(matrix,i,j-1,m,n,visited,matrix[i][j]);
        int down=dfs(matrix,i,j+1,m,n,visited,matrix[i][j]);
        visited[i][j]=1+Math.max(Math.max(l,r),Math.max(up,down));
        return visited[i][j];
    }
}
连接词
python
def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
    def check(word):
        p = trie
        for i, char in enumerate(word):
            if char not in p:
                return False
            p = p[char]
            if '#' in p and (i==len(word)-1 or check(word[i+1:])):
                return True
        return False

    T = lambda: defaultdict(T)
    res, trie = [], T()
    for word in sorted(words, key=len):
        if check(word):
            res.append(word)
        else:
            reduce(dict.__getitem__, word, trie)['#'] = {}
    return res


Java

class Trie {
    boolean end = false;
    Trie[] nexts = new Trie[26];
}

class Solution {
    Trie root = new Trie();
    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        // 按字符串的长度进行排序
        Arrays.sort(words, new Comparator<String>(){
            @Override
            public int compare(String a, String b) {
                int i = a.length(), j = b.length();
                if (i == j)
                    return 0;
                if (i > j)
                    return 1;
                else
                    return -1;
            } 
        });
        // System.out.println(Arrays.toString(words));
        List<String> ans = new ArrayList<>();
        for(String word: words) {
            if (word.length() == 0)
                continue;
            if (search(word))
                ans.add(word);
            else
                insert(word);
        }
        return ans;
    }

    public boolean search(String word){
        Trie cur = root;
        int len = word.length();
        if (len == 0)
            return true;
        for(int i = 0, ch; i < len; ++i){
            ch = word.charAt(i) - 'a';
            if (cur.nexts[ch] == null) 
                return false;
            if (cur.nexts[ch].end) {
                if (search(word.substring(i+1)))
                    return true;
            }
            cur = cur.nexts[ch];
        }
        return false;
    }
    
    public void insert(String word){
        Trie cur = root;
        for(int i = 0, len = word.length(), ch; i < len; ++i) {
            ch = word.charAt(i) - 'a';
            if (cur.nexts[ch] == null){
                cur.nexts[ch] = new Trie();
            }
            cur = cur.nexts[ch];
        }
        cur.end = true;
    }
}

回文子串

python
class Solution:
    def countSubstrings(self, s: str) -> int:
        # 以每个位置作为回文中心，尝试扩展
        # 回文中心有2种形式，1个数或2个数
        n = len(s)

        def spread(left, right):
            nonlocal ans
            while left >= 0 and right <= n - 1 and s[left] == s[right]:
                left -= 1
                right += 1
                ans += 1
    
        ans = 0
        for i in range(n):
            spread(i, i)
            spread(i, i + 1)
    
        return ans



Java

class Solution {
    int num = 0;
    public int countSubstrings(String s) {
        for (int i=0; i < s.length(); i++){
            count(s, i, i);//回文串长度为奇数
            count(s, i, i+1);//回文串长度为偶数
        }
        return num;
    }
    
    public void count(String s, int start, int end){
        while(start >= 0 && end < s.length() && s.charAt(start) == s.charAt(end)){
            num++;
            start--;
            end++;
        }
    }
}


尽量减少恶意软件的传播

python
class DSU:
    def __init__(self, N):
        self.p = range(N)
        self.sz = [1] * N

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        self.p[xr] = yr
        self.sz[yr] += self.sz[xr]
    
    def size(self, x):
        return self.sz[self.find(x)]


class Solution(object):
    def minMalwareSpread(self, graph, initial):
        dsu = DSU(len(graph))

        for j, row in enumerate(graph):
            for i in xrange(j):
                if row[i]:
                    dsu.union(i, j)
    
        count = collections.Counter(dsu.find(u) for u in initial)
        ans = (-1, min(initial))
        for node in initial:
            root = dsu.find(node)
            if count[root] == 1:  # unique color
                if dsu.size(root) > ans[0]:
                    ans = dsu.size(root), node
                elif dsu.size(root) == ans[0] and node < ans[1]:
                    ans = dsu.size(root), node
    
        return ans[1]


Java
class Solution {
    public int minMalwareSpread(int[][] graph, int[] initial) {
        int N = graph.length;
        DSU dsu = new DSU(N);
        for (int i = 0; i < N; ++i)
            for (int j = i+1; j < N; ++j)
                if (graph[i][j] == 1)
                    dsu.union(i, j);

        int[] count = new int[N];
        for (int node: initial)
            count[dsu.find(node)]++;
    
        int ans = -1, ansSize = -1;
        for (int node: initial) {
            int root = dsu.find(node);
            if (count[root] == 1) {  // unique color
                int rootSize = dsu.size(root);
                if (rootSize > ansSize) {
                    ansSize = rootSize;
                    ans = node;
                } else if (rootSize == ansSize && node < ans) {
                    ansSize = rootSize;
                    ans = node;
                }
            }
        }
    
        if (ans == -1) {
            ans = Integer.MAX_VALUE;
            for (int node: initial)
                ans = Math.min(ans, node);
        }
        return ans;
    }
}


class DSU {
    int[] p, sz;

    DSU(int N) {
        p = new int[N];
        for (int x = 0; x < N; ++x)
            p[x] = x;
    
        sz = new int[N];
        Arrays.fill(sz, 1);
    }
    
    public int find(int x) {
        if (p[x] != x)
            p[x] = find(p[x]);
        return p[x];
    }
    
    public void union(int x, int y) {
        int xr = find(x);
        int yr = find(y);
        p[xr] = yr;
        sz[yr] += sz[xr];
    }
    
    public int size(int x) {
        return sz[find(x)];
    }
}



颜色交替最短路径

python 
class Solution:
    def shortestAlternatingPaths(
        self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]
    ) -> List[int]:
        g = [defaultdict(list), defaultdict(list)]
        for i, j in redEdges:
            g[0][i].append(j)
        for i, j in blueEdges:
            g[1][i].append(j)
        ans = [-1] * n
        vis = set()
        q = deque([(0, 0), (0, 1)])
        d = 0
        while q:
            for _ in range(len(q)):
                i, c = q.popleft()
                if ans[i] == -1:
                    ans[i] = d
                vis.add((i, c))
                c ^= 1
                for j in g[c][i]:
                    if (j, c) not in vis:
                        q.append((j, c))
            d += 1
        return ans


Java
class Solution {
    public int[] shortestAlternatingPaths(int n, int[][] redEdges, int[][] blueEdges) {
        List<Integer>[][] g = new List[2][n];
        for (var f : g) {
            Arrays.setAll(f, k -> new ArrayList<>());
        }
        for (var e : redEdges) {
            g[0][e[0]].add(e[1]);
        }
        for (var e : blueEdges) {
            g[1][e[0]].add(e[1]);
        }
        Deque<int[]> q = new ArrayDeque<>();
        q.offer(new int[] {0, 0});
        q.offer(new int[] {0, 1});
        boolean[][] vis = new boolean[n][2];
        int[] ans = new int[n];
        Arrays.fill(ans, -1);
        int d = 0;
        while (!q.isEmpty()) {
            for (int k = q.size(); k > 0; --k) {
                var p = q.poll();
                int i = p[0], c = p[1];
                if (ans[i] == -1) {
                    ans[i] = d;
                }
                vis[i][c] = true;
                c ^= 1;
                for (int j : g[c][i]) {
                    if (!vis[j][c]) {
                        q.offer(new int[] {j, c});
                    }
                }
            }
            ++d;
        }
        return ans;
    }
}


统计所有可行路径

python
class Solution:
    def countRoutes(self, locations: List[int], start: int, finish: int, fuel: int) -> int:
        dp = [[0] * len(locations) for _ in range(fuel + 1)]
        dp[fuel][start] = 1
        diff = 0
        for i in range(fuel, 0, -1):
            for src in range(len(locations)):
                for dst in range(len(locations)):
                    if src == dst:
                        continue
                    diff = abs(locations[src] - locations[dst])
                    if i >= diff:
                        dp[i - diff][dst] += dp[i][src]
        ans = 0
        for i in range(len(dp)):
            ans += dp[i][finish]
        return ans % (10 ** 9 + 7)



Java
class Solution {
    private static final int MOD = (int)(1e9+7);
    public int countRoutes(int[] locations, int start, int finish, int fuel) {
        final int n = locations.length;
        int[][] dp = new int[fuel + 1][n];
        dp[fuel][start] = 1;
        int[][] diff = new int[n][n];
        for (int i = 0 ; i < n; ++i){
            for (int j = i + 1; j < n; ++j){
                diff[i][j] = Math.abs(locations[i] - locations[j]);
                diff[j][i] = diff[i][j];
            }
        }
        for (int i = fuel; i >= 0; --i){
            for (int src = 0; src < n; ++src){
                for (int des = 0; des < n; ++des){
                    if (src == des){
                        continue;
                    }
                    if (i >= diff[src][des]){
                        dp[i - diff[src][des]][des] += dp[i][src];
                        if (dp[i - diff[src][des]][des] >= MOD){
                            dp[i - diff[src][des]][des] -= MOD;
                        }
                    }
                }
            }
        }
        int ans = 0;
        for (int i = 0; i <= fuel; ++i){
            ans += dp[i][finish];
            if (ans >= MOD){
                ans -= MOD;
            }
        }
        return ans;
    }
}


快速公交

python
class Solution:
    def busRapidTransit(self, target: int, inc: int, dec: int, jump: List[int], cost: List[int]) -> int:
        deq = [[0, target]]#最小堆
        while deq:
            i, t = heapq.heappop(deq)
            if t == 0:
                return i % (10 ** 9 + 7)
            heapq.heappush(deq, [i + t * inc, 0])#直接走到起点
            for j, n in enumerate(jump):
                mod = t % n
                heapq.heappush(deq, [i + mod * inc + cost[j], t // n])#向前走
                heapq.heappush(deq, [i + (n - mod) * dec + cost[j], t // n + 1])向后走


Java

public class Solution {

  private static final int MOD = 1000000007;

  private int inc;

  private int dec;

  private int[] jump;

  private int[] cost;

  private Map<Long, Long> memo = new HashMap<>();

  public int busRapidTransit(int target, int inc, int dec, int[] jump, int[] cost) {
    this.inc = inc;
    this.dec = dec;
    this.jump = jump;
    this.cost = cost;
    return (int) (busRapidTransitHelper(target) % MOD);
  }

  private long busRapidTransitHelper(long target) {
    if (target == 0) {
      return 0;
    }
    Long result = memo.get(target);
    if (null != result) {
      return result;
    }
    result = target * 1L * inc;
    for (int i = 0; i < jump.length; i++) {
      result = Math.min(result, cost[i] + (target % jump[i]) * 1L * inc + busRapidTransitHelper(target / jump[i]));
      if (target > 1 && (target % jump[i]) != 0) {
        result = Math.min(result, cost[i] + (jump[i] - (target % jump[i])) * 1L * dec + busRapidTransitHelper(target / jump[i] + 1));
      }
    }
    memo.put(target, result);
    return result;
  }

}


访问所有节点的最短路径
python
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        q = deque((i, 1 << i, 0) for i in range(n))
        seen = {(i, 1 << i) for i in range(n)}
        ans = 0
        
        while q:
            u, mask, dist = q.popleft()
            if mask == (1 << n) - 1:
                ans = dist
                break
            # 搜索相邻的节点
            for v in graph[u]:
                # 将 mask 的第 v 位置为 1
                mask_v = mask | (1 << v)
                if (v, mask_v) not in seen:
                    q.append((v, mask_v, dist + 1))
                    seen.add((v, mask_v))
        
        return ans

class Solution {
    public int shortestPathLength(int[][] graph) {
        int n = graph.length;
        Queue<int[]> queue = new LinkedList<int[]>();
        boolean[][] seen = new boolean[n][1 << n];
        for (int i = 0; i < n; ++i) {
            queue.offer(new int[]{i, 1 << i, 0});
            seen[i][1 << i] = true;
        }

        int ans = 0;
        while (!queue.isEmpty()) {
            int[] tuple = queue.poll();
            int u = tuple[0], mask = tuple[1], dist = tuple[2];
            if (mask == (1 << n) - 1) {
                ans = dist;
                break;
            }
            // 搜索相邻的节点
            for (int v : graph[u]) {
                // 将 mask 的第 v 位置为 1
                int maskV = mask | (1 << v);
                if (!seen[v][maskV]) {
                    queue.offer(new int[]{v, maskV, dist + 1});
                    seen[v][maskV] = true;
                }
            }
        }
        return ans;
    }
}

买卖股票的最佳时机 II

python


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        dp = [[0]*2 for _ in range(len(prices))]

        for i in range(len(prices)):
            if i-1<0:
                dp[i][0] = -prices[0]
                dp[i][1] = 0
                continue
            dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i])
    
        return dp[-1][1]






Java
class Solution {
    public int maxProfit(int[] prices) {
        int ans = 0;
        int n = prices.length;
        for (int i = 1; i < n; ++i) {
            ans += Math.max(0, prices[i] - prices[i - 1]);
        }
        return ans;
    }
}


最小覆盖字串

python
    def minWindow(self, s: str, t: str) -> str:
        need=collections.defaultdict(int)
        for c in t:
            need[c]+=1
        needCnt=len(t)
        i=0
        res=(0,float('inf'))
        for j,c in enumerate(s):
            if need[c]>0:
                needCnt-=1
            need[c]-=1
            if needCnt==0:       #步骤一：滑动窗口包含了所有T元素
                while True:      #步骤二：增加i，排除多余元素
                    c=s[i] 
                    if need[c]==0:
                        break
                    need[c]+=1
                    i+=1
                if j-i<res[1]-res[0]:   #记录结果
                    res=(i,j)
                need[s[i]]+=1  #步骤三：i增加一个位置，寻找新的满足条件滑动窗口
                needCnt+=1
                i+=1
        return '' if res[1]>len(s) else s[res[0]:res[1]+1]    #如果res始终没被更新过，代表无满足条件的结果

class Solution {
public:
    unordered_map <char, int> ori, cnt;

    bool check() {
        for (const auto &p: ori) {
            if (cnt[p.first] < p.second) {
                return false;
            }
        }
        return true;
    }
    
    string minWindow(string s, string t) {
        for (const auto &c: t) {
            ++ori[c];
        }
    
        int l = 0, r = -1;
        int len = INT_MAX, ansL = -1, ansR = -1;
    
        while (r < int(s.size())) {
            if (ori.find(s[++r]) != ori.end()) {
                ++cnt[s[r]];
            }
            while (check() && l <= r) {
                if (r - l + 1 < len) {
                    len = r - l + 1;
                    ansL = l;
                }
                if (ori.find(s[l]) != ori.end()) {
                    --cnt[s[l]];
                }
                ++l;
            }
        }
    
        return ansL == -1 ? string() : s.substr(ansL, len);
    }
};


无重复字符的最长子串
python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:return 0
        left = 0
        lookup = set()
        n = len(s)
        max_len = 0
        cur_len = 0
        for i in range(n):
            cur_len += 1
            while s[i] in lookup:
                lookup.remove(s[left])
                left += 1
                cur_len -= 1
            if cur_len > max_len:max_len = cur_len
            lookup.add(s[i])
        return max_len

Java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s.length()==0) return 0;
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        int max = 0;
        int left = 0;
        for(int i = 0; i < s.length(); i ++){
            if(map.containsKey(s.charAt(i))){
                left = Math.max(left,map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i),i);
            max = Math.max(max,i-left+1);
        }
        return max;
        
    }
}


字母异位词分组

python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)

        for st in strs:
            counts = [0] * 26
            for ch in st:
                counts[ord(ch) - ord("a")] += 1
            # 需要将 list 转换成 tuple 才能进行哈希
            mp[tuple(counts)].append(st)
        
        return list(mp.values())

java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for (String str : strs) {
            int[] counts = new int[26];
            int length = str.length();
            for (int i = 0; i < length; i++) {
                counts[str.charAt(i) - 'a']++;
            }
            // 将每个出现次数大于 0 的字母和出现次数按顺序拼接成字符串，作为哈希表的键
            StringBuffer sb = new StringBuffer();
            for (int i = 0; i < 26; i++) {
                if (counts[i] != 0) {
                    sb.append((char) ('a' + i));
                    sb.append(counts[i]);
                }
            }
            String key = sb.toString();
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }
}


二叉树着色
class Solution:
    def btreeGameWinningMove(self, root: TreeNode, n: int, x: int) -> bool:
        xNode = None

        def getSubtreeSize(node):
            if not node:
                return 0
            if node.val == x:
                nonlocal xNode
                xNode = node
            return 1 + getSubtreeSize(node.left) + getSubtreeSize(node.right)
    
        getSubtreeSize(root)
        leftSize = getSubtreeSize(xNode.left)
        if leftSize >= (n + 1) // 2:
            return True
        rightSize = getSubtreeSize(xNode.right)
        if rightSize >= (n + 1) // 2:
            return True
        remain = n - leftSize - rightSize - 1
        return remain >= (n + 1) // 2




打家劫舍3

class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode]) -> (int, int):
            if node is None:  # 递归边界
                return 0, 0  # 没有节点，怎么选都是 0
            l_rob, l_not_rob = dfs(node.left)  # 递归左子树
            r_rob, r_not_rob = dfs(node.right)  # 递归右子树
            rob = l_not_rob + r_not_rob + node.val  # 选
            not_rob = max(l_rob, l_not_rob) + max(r_rob, r_not_rob)  # 不选
            return rob, not_rob
        return max(dfs(root))  # 根节点选或不选的最大值


主体空间

class Solution:
    def largestArea(self, grid: List[str]) -> int:
        
        def dfs(grid, r, c, sig):
            if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                return -1
            if grid[r][c] == '0':
                return -1
            if grid[r][c] != sig:
                return 0
    
            grid[r][c] = '-1'
            
            a1 = dfs(grid, r - 1, c, sig)
            a2 = dfs(grid, r + 1, c, sig)
            a3 = dfs(grid, r, c - 1, sig)
            a4 = dfs(grid, r, c + 1, sig)
    
            if a1 != -1 and a2 != -1 and a3 != -1 and a4 != -1: 
                return 1 + a1 + a2 + a3 + a4
            else:
                return -1
        
        grid = [list(grid[idx]) for idx in range(len(grid))]
        
        ans = 0
        nrow, ncol = len(grid), len(grid[0])
        for irow in range(nrow):
            for icol in range(ncol):
                if grid[irow][icol] != '0' and grid[irow][icol] != '-1':
                    ans = max(dfs(grid, irow, icol, grid[irow][icol]), ans)
        return ans


找出知晓秘密的所有专家

class Solution:
    def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
        m = len(meetings)
        meetings.sort(key=lambda x: x[2])

        secret = [False] * n
        secret[0] = secret[firstPerson] = True
    
        i = 0
        while i < m:
            # meetings[i .. j] 为同一时间
            j = i
            while j + 1 < m and meetings[j + 1][2] == meetings[i][2]:
                j += 1
    
            vertices = set()
            edges = defaultdict(list)
            for k in range(i, j + 1):
                x, y = meetings[k][0], meetings[k][1]
                vertices.update([x, y])
                edges[x].append(y)
                edges[y].append(x)
            
            q = deque([u for u in vertices if secret[u]])
            while q:
                u = q.popleft()
                for v in edges[u]:
                    if not secret[v]:
                        secret[v] = True
                        q.append(v)
            
            i = j + 1
        
        ans = [i for i in range(n) if secret[i]]
        return ans


被围绕的区域

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        if not board:
            return
        
        n, m = len(board), len(board[0])
    
        def dfs(x, y):
            if not 0 <= x < n or not 0 <= y < m or board[x][y] != 'O':
                return
            
            board[x][y] = "A"
            dfs(x + 1, y)
            dfs(x - 1, y)
            dfs(x, y + 1)
            dfs(x, y - 1)
        
        for i in range(n):
            dfs(i, 0)
            dfs(i, m - 1)
        
        for i in range(m - 1):
            dfs(0, i)
            dfs(n - 1, i)
        
        for i in range(n):
            for j in range(m):
                if board[i][j] == "A":
                    board[i][j] = "O"
                elif board[i][j] == "O":
                    board[i][j] = "X"



搜索二维矩阵
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        l, r = 0, m * n - 1
        while l <= r:
            mid = (l + r) >> 1
            x, y = mid // n , mid % n
            if matrix[x][y] > target:
                r = mid - 1
            elif matrix[x][y] < target:
                l = mid + 1
            else:
                return True
        return False