package Base;

import Base.Definition.ListNode;

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

    //31. 下一个排列
    public void nextPermutation(int[] nums) {
        //实话说，这个鬼题目我都没怎么看懂

        //没有太多的思路，看来题解只是觉得巧妙的很

        //题解答题的思路：从右边开始遍历数组，遇到第一个下降的元素（i-1），
        // 在[i,length-1]这一段范围进行搜索，找到最小的并且大于i-1下标元素的元素(t)
        //交换i-1和t，交换后，[i, length-1]是一段上升区间，翻转即可
        //temp变量用于记录上一个值的大小, index用于记录第一个下降的下标
        int i = nums.length - 2;
        while(i > 0 && nums[i] >= nums[i+1]){
            i--;
        }
        if(i >= 0){
            int j = nums.length - 1;
            while(j > 0 && nums[j] <= nums[i]){
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i+1);
    }

    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private void reverse(int[] nums, int start){
        int i = start, j = nums.length;
        while(i < j){
            swap(nums, i, j);
            i++;
            j--;
        }
    }
}
