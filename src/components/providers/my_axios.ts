import Axios from 'axios';

/**
 * axios wrapper https://github.com/axios/axios
 */
class AbAxios {

    public _abAxios = Axios.create();

    constructor(baseUrl: string = window.location.origin, contentType: string = "application/json; charset=utf-8") {
        this._abAxios.defaults.baseURL = baseUrl;
        this._abAxios.defaults.headers.common['Content-Type'] = contentType;
        //Todo: this._abAxios.defaults.headers.common['Authorization'] = AUTH_TOKEN
    }

    /**
     * Encode params from key value object
     * @param paramsObj
     */
    public encodeParams(paramsObj: any, isbeginQueryString: boolean): string {
        let queryString = Object.keys(paramsObj).map(function (key) {
            return [key, paramsObj[key]].map(encodeURIComponent).join("=");
        }).join("&");

        return isbeginQueryString === true ? "?"+queryString : "&"+queryString;
    }

  /**
   * Get request
   * @param urlEndpoint
   * @param paramsObj
   * @param toggleSpinner
   * @param successCallback
   * @param errorCallback
   */
    public async abGet(urlEndpoint: string, toggleSpinner: boolean, successCallback: any, errorCallback: any) {
        var self = this;

        await self._abAxios.get(urlEndpoint)
            .then(function (response) {
                successCallback(response);
            })
            .catch(function (error) {
                errorCallback(error);
            });
    }

    public async abPost(urlEndpoint: string, payload: any, contentType: string = "application/json", toggleSpinner: boolean, successCallback: any, errorCallback: any) {
        var self = this;

        await self._abAxios.post(urlEndpoint, payload, {
            headers: {
                'Content-Type': contentType
            }
        })
            .then(function (response) {
                successCallback(response);
            })
            .catch(function (error) {
                errorCallback(error);
            });
    }
}

export default AbAxios;
