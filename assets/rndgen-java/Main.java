import java.io.BufferedReader;
import java.io.IOException;
import java.math.BigInteger;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws IOException {

        Random random = new Random();

        //treat the first arg (if any) as seed
        if (args.length > 0) {
            try {
                int seed = Integer.parseInt(args[0]);
                random = new Random(seed);
            } catch (NumberFormatException e) {
            }
        }

        byte[] buff = new byte[1024];

        while(true) {
            random.nextBytes(buff);
            System.out.write(buff);
            System.out.flush();
        }
    }

}

