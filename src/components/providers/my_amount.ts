class AbAmount {


    objFormat: any = { "locale": "da", "maxDigits": 0 };

    constructor() {

    }



    format(value: number) {
        if (value === null) {
            return 0;
        }
        let formattedValue = value.toLocaleString(this.objFormat.locale, { maximumFractionDigits: this.objFormat.maxDigits });
        return formattedValue;
    }

    /*format(value: number, objFormat: any) {
        let formattedValue = value.toLocaleString(objFormat.locale, { maximumFractionDigits: objFormat.maxDigits });
        return formattedValue;
    }*/
}

export default AbAmount;
