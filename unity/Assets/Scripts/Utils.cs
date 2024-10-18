using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class Utils
{
    static DateTime beginOfUniverse = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
    public static long GetUnixTime()
    {
        return (long)(DateTime.UtcNow - beginOfUniverse).TotalMilliseconds;
    }
}
