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
}
