package Base;

import Base.Definition.ListNode;
import Base.Definition.TreeNode;

import java.util.*;

/**
 * @ClassName: Solution
 * @Description:
 * @author: SeanLeong
 * @date: 2020/10/9 9:53
 */
public class Solution {

    //第141题目：环形列表
    public boolean hasCycle(ListNode head) {
        //使用集合，已经访问过的放入set就可以了

        //进阶：使用O(1)的空间复杂度,这边使用快慢指针

        //头节点判断
        if(head == null || head.next == null){
            return false;
        }

        //快慢指针
        ListNode fast = head.next;
        ListNode slow = head;

        //循环
        while(slow != fast){
            if(fast == null || fast.next == null)
                return false;
            slow = slow.next;
            fast = fast.next.next;
        }

        return true;
    }

    //933.最近请求次数
    public static class RecentCounter {

        int recentTime; //记录最近ping的时间
        Queue<Integer> queue = new LinkedList<>();

        public RecentCounter() {
            recentTime = 0;
        }

        public int ping(int t) {
            if(t < recentTime){
                throw new RuntimeException();
            }
            //修改最近的ping的时间
            recentTime = t;

           queue.add(t);

           while(queue.peek() != null && t - queue.peek() > 3000){
                   queue.poll();
           }

           return queue.size();
        }
    }

    //1.两数之和
    public int[] twoSum(int[] nums, int target) {
        //可以转成map，然后使用map进行计算，k为num,v为index
        Map<Integer, Integer> map = new HashMap<>();

        //循环数组，转为map
        for(int i = 0; i < nums.length; i++){
            if(map.containsKey(target - nums[i])){
                return new int[]{map.get(target - nums[i]), i};
            }
            map.put(nums[i], i);
        }
        return new int[0];
    }

    //11. 盛最多水的容器
    public int maxArea(int[] height) {
        //双指针解法：使用双指针的时候需要考虑的事情是：移动指针，如何移动

        //移动值小的那个，如果相等规定移动左边的

        //定义两个指针，左右指针，以及最大容积
        int left = 0, right = height.length - 1, maxArea = 0, currArea;

        //双指针一般都使用while循环作为终止条件

        while(left <= right){
            currArea = Math.min(height[left], height[right]) * (right - left);
            maxArea = Math.max(maxArea, currArea);
            if(height[right] < height[left]){
                right--;
            }else{
                left++;
            }
        }
        return maxArea;
    }

    //15. 三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        /*
            刚刚看了题解，暴力的解法是不会考虑的，
            题解思路：排序，双重循环，双指针
        */
        List<List<Integer>> ans = new ArrayList<>();
        if(nums == null || nums.length < 3){
            return ans;
        }
        //排序
        Arrays.sort(nums);


        for(int i = 0; i < nums.length - 2; i++){
            //如果与前面的相等，我们就没有必要重复了（防止重复）
            if(i > 0 && nums[i] == nums[i-1])
                continue;
            //k与j形成一个双指针
            int k = nums.length - 1;
            //目标数
            int target = -nums[i];
            for(int j = i+1; j < nums.length; j++){
                //同理（防止重复）
                if(j > i+1 && nums[j] == nums[j - 1])
                    continue;
                while(j <  k && nums[j] + nums[k] > target){
                    k--;
                }
                //相等说明他们直接不可能了，退出
                if(j == k){
                    break;
                }
                //没有相等，说明存在，添加经ans里头
                if(nums[j] + nums[k] == target) {
                    List<Integer> ans_item = Arrays.asList(nums[i], nums[j], nums[k]);
                    ans.add(ans_item);
                }
            }
        }
        return ans;
    }

    //530. 二叉搜索树的最小绝对差
    public int getMinimumDifference(TreeNode root) {
        //二叉搜索树，中序遍历会得到一个排好序的
        List<Integer> list = new ArrayList<>();
        inorder(root, list);
        //此时list已经排序成功，相邻的节点差最小，所以做遍历即可
        int min = Integer.MAX_VALUE;
        for(int i=0; i < list.size() - 1 ; i++){
            min = Math.min(min ,Math.abs(list.get(i) - list.get(i+1)));
        }
        return min;
    }

    //中序
    public void inorder(TreeNode root, List<Integer> result){

        /*
        递归实现
        if(root == null){
            return;
        }
        inorder(root.left, result);
        result.add(root.val);
        inorder(root.right, result);

         */

        //栈实现
        Stack<TreeNode> stack = new Stack<>();
        while(root != null || !stack.empty()){
            //节点存在、或者栈非空

            //节点存在就添加节点
            while(root != null){
                stack.push(root);
                root = root.left;
            }
            //此时root == null ,
            // 栈非空就弹出节点，并加入他的右节点
            root = stack.pop();
            result.add(root.val);
            root = root.right;
        }
    }

    //TODO:【非最优解】94. 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        //使用递归
        List<Integer> result = new ArrayList<>();
        inorder(root, result);
        return result;
    }

    //31. 下一个排列
    public void nextPermutation(int[] nums) {
        //实话说，这个鬼题目我都没怎么看懂

        //没有太多的思路，看来题解只是觉得巧妙的很

        //题解答题的思路：从右边开始遍历数组，遇到第一个下降的元素（i-1），
        // 在[i,length-1]这一段范围进行搜索，找到最小的并且大于i-1下标元素的元素(t)
        //交换i-1和t，交换后，[i, length-1]是一段上升区间，翻转即可
        //temp变量用于记录上一个值的大小, index用于记录第一个下降的下标
        int i = nums.length - 2;
        while (i >= 0 && nums[i + 1] <= nums[i]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private void reverse(int[] nums, int start){
        int i = start, j = nums.length-1;
        while(i < j){
            swap(nums, i, j);
            i++;
            j--;
        }
    }
    //16.最接近的三数之和
    public int threeSumClosest(int[] nums, int target) {
        int best = Integer.MAX_VALUE;
        int n = nums.length;
        //先排序、第一个数字使用循环即可，第二第三个数字我们使用双指针策略
        Arrays.sort(nums);
        for(int i=0; i<n-2; i++){
            //在第一层就不能重复了
            if(i != 0 && nums[i] == nums[i-1]){
                continue;
            }
            int j = i + 1, k = n-1;
            while( j < k){
                int sum = nums[i] + nums[j] + nums[k];
                if(sum == target)
                    return target;
                if(Math.abs(sum - target) < Math.abs(best - target)){
                    best = sum;
                }
                if(sum > target){
                    k--;
                }else{
                    j++;
                }
            }
        }
        return best;
    }

    //24. 两两交换链表中的节点
    public ListNode swapPairs(ListNode head) {
        if(head == null || head.next == null){
            return head;
        }

        //有点想不出怎么做了，不想浪费太多时间，看了一下那个题解

        /*
            1.方法一：递归
            使用递归将后面的节点都两两交换
            然后改变当前两个节点的位置即可

            ListNode newHead = head.next;
            //后面的节点先进行交换
            //head指向后面节点的头节点
            head.next = swapPairs(newHead.next);
            newHead.next = head;
            return newHead;
         */

        /*其实简单的循环也可以完成


         */
        //指向头节点的哑节点
        ListNode dummyHead = new ListNode(0, head);
        ListNode pre = dummyHead;
        ListNode after;
        while(pre.next != null && pre.next.next != null){

            after = pre.next.next;
            head = pre.next;

            //交换
            head.next = head.next.next;
            after.next = head;
            pre.next = after;

            //移动pre/head/after
            pre = head;

        }
        return dummyHead.next;
    }

    //21. 合并两个有序链表
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null){
            return l2;
        }else if(l2 == null){
            return l1;
        }
//        else if(l1.val < l2.val){
//            l1.next = mergeTwoLists(l1.next, l2);
//            return l1;
//        }else{
//            l2.next = mergeTwoLists(l1, l2.next);
//            return l2;
//        }
        //也可以使用递归的方式进行求解，上面就行


        //下面的空间复杂度是o(n)可以改成o(1)，直接改变指针即可，不难
        ListNode head = new ListNode();
        ListNode temp = head;
        while(l1 != null && l2 != null){
            //因为是升序，我们保存小的作为插入的节点
//            node = node.next;
            ListNode node = new ListNode();
            if(l1.val < l2.val){
                node = l1;
                l1 = l1.next;
            }else{
                node = l2;
                l2 = l2.next;
            }
            temp.next = node;
            temp = node;
        }
        while(l1 != null){
            ListNode node = new ListNode();
            node = l1;
            l1 = l1.next;
            temp.next = node;
            temp = node;
        }
        while(l2 != null){
            ListNode node = new ListNode();
            node = l2;
            l2 = l2.next;
            temp.next = node;
            temp = node;
        }
        return head.next;
    }

    //27. 移除元素
    public int removeElement(int[] nums, int val) {
        if(nums.length == 0){
            return 0;
        }

        //使用双指针
//        int slow = 0;
//        for(int i=0; i<nums.length; i++){
//            if(nums[i] != val){
//                nums[slow] = nums[i];
//                slow++;
//            }
//        }
//        return slow;

        /*
            刚刚看了题解，发现还可以是进行交换：
                将想要移出的元素跟数组中最后面的元素进行交换，然后将减少数组的长度
                但是交换后，当前元素的下标保持不变。
                如果当前元素与移除元素不相等的话，那么我们就移动当前元素的下标
         */
        int i = 0; //表示当前元素的下标
        int n = nums.length;
        while(i < n){
            if( nums[i] == val){
                nums[i] = nums[n-1];
                n--;
                //此时当前元素的节点不改变
            } else {
                i++;
            }
        }
        return n;

    }

    //28. 实现 strStr()//TODO：KMP的实现
    public int strStr(String haystack, String needle) {
        if("".equals(needle) || null == needle){
            return 0;
        }
        //kmp算法,不会写emm

        return 0;
    }

    //33. 搜索旋转排序数组
    public int search(int[] nums, int target) {
        /*
            思路：二分法，分开两部分，一般有序另一边则无序，
         */

        int left = 0, right = nums.length - 1;
        while(left <= right){
            int mid = (left + right) / 2;
            if(nums[mid] == target){
                return mid;
            }
            if(nums[mid] <= nums[right]){
                //说明旋转点在mid之前或等于
                if(target >= nums[mid] && target <= nums[right]){
                    left = mid;
                }else{
                    right = mid-1;
                }
            }else{
                //旋转点在中点之后
                if(target <= nums[mid] && target >= nums[left]){
                    right = mid;
                }else{
                    left = mid+1;
                }
            }
        }
        return -1;
    }

//    34. 在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        /*
            题目要求是logN的事件复杂度
            二分查找没得选了,我的思路是找到对应的target下边后，从两个方向扩散，找到最开始和最后结束的
         */
        /*
        int n = nums.length;
        int[] result = {-1, -1};
        if(n == 0 || target < nums[0] || target > nums[n-1]){ //优化一波
            return result;
        }
        int r = 0, l = n - 1;
        int index = -1, start = -1, end = -1;
        while(r <= l){
            int mid = (r + l) / 2;
            if(nums[mid] == target){
                index = mid;
                break;
            }else if(nums[mid] > target){
                l = mid - 1;
            }else{
                r = mid + 1;
            }
        }
        if(index != -1){
            start = index;
            end = index;
            while(start >= 0 && nums[start] == target){
                start--;
            }
            while(end < n && nums[end] == target){
                end++;
            }
            result[0] = start+1;
            result[1] = end-1;
        }
        return result;
         */

        int n = nums.length;
        int[] result = {-1, -1};
        if(n == 0 || target < nums[0] || target > nums[n-1]){ //优化一波
            return result;
        }

        result[0] = getFirst(nums, target);
        result[1] = getLast(nums, target);
        return result;
    }

    private int getFirst(int nums[], int target){
        int r = 0, l = nums.length - 1;
        boolean flag = false;
        while(r <= l){
            int mid = (r + l) / 2;
            if(nums[mid] == target){
                flag = true;
            }
            if(target > nums[mid]){
                r = mid + 1;
            }else{
                l = mid - 1;
            }
        }
        if(flag) {
            return r;  //因为相等的时候会走else，改变的是l
        }else{
            return -1;
        }
    }
    private int getLast(int nums[], int target){
        int r = 0, l = nums.length - 1;
        boolean flag = false;
        while(r <= l){
            int mid = (r + l) / 2;
            if(nums[mid] == target){
                flag = true;
            }
            if(target < nums[mid]){
                l = mid - 1;
            }else{
                r = mid + 1;
            }
        }
        if(flag) {
            return l;
        }else{
            return -1;
        }
    }

    //925. 长按键入
    public boolean isLongPressedName(String name, String typed) {
        if("".equals(name) || "".equals(typed) || name.length() > typed.length()){
            return false;
        }

        int i = 0, j = 0;
        while(j < typed.length()){
            if(i < name.length() && name.charAt(i) == typed.charAt(j)){
                i++;
                j++;
            }else if(j > 0 && typed.charAt(j-1) == typed.charAt(j)){//此时字符不相等、判断是否有重复，重复了则下一个
                j++;
            }else{
                return false;
            }
        }
        return i == name.length();
    }

    //35. 搜索插入位置
    public int searchInsert(int[] nums, int target) {
        if(nums.length == 0){
            return 0;
        }

        int l = 0, r = nums.length-1;
        while(l < r){
            int mid = (l + r) / 2;
            if(nums[mid] == target){
                return mid;
            }
            if(target > nums[mid]){
                l = mid + 1;
            }else{
                r = mid - 1;
            }
        }
        if(nums[l] < target){
            return l+1;
        }
        return l;
    }

    //36. 有效的数独
    public boolean isValidSudoku(char[][] board) {
        boolean[][] row = new boolean[9][9];
        boolean[][] col = new boolean[9][9];
        boolean[][] block = new boolean[9][9];

        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                if(board[i][j] != '.'){
                    int index = board[i][j] - '1';
                    int blockIndex = (i / 3) * 3 + (j / 3);
                    if(row[i][index] || col[j][index] || block[blockIndex][index]){
                        return false;
                    }else{
                        row[i][index] = true;
                        col[j][index] = true;
                        block[blockIndex][index] = true;
                    }
                }
            }
        }
        return true;
    }


    //40. 组合总和 II
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        //有重复元素，我的思路是直接排序，然后在进行搜索
        Arrays.sort(candidates);
        dfsCombinationSum2(result, candidates, new ArrayList<>(), target, 0);
        return result;
    }

    private void dfsCombinationSum2(List<List<Integer>> result, int[] candidates, List<Integer> path,
                                   int target, int index){
        if(target == 0){
            result.add(new ArrayList<>(path));
            return ;
        }
        if(path.size() == candidates.length){
            return ;
        }

        for(int i = index; i < candidates.length; i++){
            if(target - candidates[i] >= 0){
                //要去重的话，需要判断当前层，不能重复；
                if( i > index && candidates[i] == candidates[i-1]){
                    continue;
                }
                path.add(candidates[i]);
                dfsCombinationSum2(result, candidates, path, target - candidates[i], i+1);
                path.remove(path.size() - 1);
            }
        }
    }
}
















