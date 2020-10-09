package Base;

/**
 * @ClassName: Test
 * @Description:
 * @author: SeanLeong
 * @date: 2020/10/9 9:52
 */
public class Test {
    public static void main(String[] args) {
        Solution solution = new Solution();
        int[]  nums = {2, 7, 11, 15};
        int target = 9;
        System.out.println(solution.twoSum(nums, target)[0]);
        System.out.println(solution.twoSum(nums, target)[1]);
    }
}
